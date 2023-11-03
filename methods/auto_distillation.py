import copy
import random
import torch
import torch.nn.functional as F

from models import CLIP_MODELS
from methods.source_only import SourceOnly
from tools.utils import get_save_logits_dir, get_save_checkpoint_dir, get_save_scores_dir

"""
This is the mehod proposed in "Universal Domain Adaptation from Foundation Models: A Baseline Study"
"""
class AutoDistill(SourceOnly):
    require_source = False
    require_target = True
    _hyparas = {'source_black_box': 
                {'office31': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'officehome': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'visda': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'domainnet': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}'}
                }
    # threshold_mode = 'ENTROPY'

    def __init__(self, cfg) -> None:
        self.in_clip = cfg.backbone in CLIP_MODELS
        assert self.in_clip
        if self.in_clip:
            cfg.fixed_backbone = True
            black_box_backbone = cfg.backbone.replace('/', '') # 'ViT-L14@336px'
            black_box_method = 'ClipZeroShot'
            seed = 1

        cfg.classifier_head = 'prototype'
        
        super().__init__(cfg)
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.0)

        file_path = get_save_logits_dir(cfg.feature_dir, 
                                        self._hyparas['source_black_box'][cfg.dataset].format(black_box_method, black_box_backbone, cfg.max_iter), 
                                        cfg.dataset, 
                                        cfg.source_domain, 
                                        cfg.target_domain, 
                                        cfg.n_share, 
                                        cfg.n_source_private, 
                                        seed)
        
        try:
            self.target_logits = torch.load(file_path).to(self.device)
        except:
            scores_pth = get_save_scores_dir(cfg.feature_dir, 
                                        self._hyparas['source_black_box'][cfg.dataset].format(black_box_method, black_box_backbone, cfg.max_iter), 
                                        cfg.dataset, 
                                        cfg.source_domain, 
                                        cfg.target_domain, 
                                        cfg.n_share, 
                                        cfg.n_source_private, 
                                        seed,
                                        prefix='scores')
            self.target_logits = torch.load(scores_pth)['target_logits'].to(self.device)

        self.logits, self.pseudo_labels = torch.max(self.target_logits, dim=-1)

        self.file_path_checkpoint = get_save_checkpoint_dir(cfg.feature_dir, 
                                                       self._hyparas['source_black_box'][cfg.dataset].format(black_box_method, black_box_backbone, cfg.max_iter), 
                                                       cfg.dataset, 
                                                       cfg.source_domain, 
                                                       cfg.target_domain, 
                                                       cfg.n_share, 
                                                       cfg.n_source_private, 
                                                       seed)

    
    def before_training(self, cfg=None):
        super().before_training(cfg)
        
        state_dict = torch.load(self.file_path_checkpoint)
        self.load_state_dict(state_dict, strict=False)
        self.prototypes = copy.deepcopy(self.classifier.fc.weight.data)

        if isinstance(cfg, dict):
            source_dataloader = cfg['val_data_loader']
            self.set_temperature(source_dataloader)
    
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)
        ece_criterion_ood = _ECELossOOD().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch_datas in valid_loader:
                input, label = batch_datas['img'], batch_datas['label']
                input = input.to(self.device)
                # logits = self.model(input)
                features = self.backbone(input)
                logits = self.classifier(features)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)
        
        # devide logits to in and out samples
        num_classes = labels.max().item()//2
        mask = labels < num_classes
        logits_in = logits[mask]
        logits_in = logits_in[:,:num_classes]
        logits_out = logits[~mask]
        logits_out = logits_out[:,:num_classes]
        labels_in = labels[mask]
        labels_out = labels[~mask]

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits_in, labels_in).item()
        before_temperature_ece_out = ece_criterion_ood(logits_out, num_classes).item()
        before_temperature_ece = ece_criterion(logits_in, labels_in).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f, ECE_OUT: %.3f' % (before_temperature_nll, before_temperature_ece, before_temperature_ece_out))

        def eval():
            optimizer.zero_grad()
            # loss = nll_criterion(self.temperature_scale(logits_in), labels_in) + ece_criterion_ood(self.temperature_scale(logits_out), num_classes)
            loss = nll_criterion(self.temperature_scale(logits_in), labels_in) + ece_criterion(self.temperature_scale(logits_in), labels_in) + ece_criterion_ood(self.temperature_scale(logits_out), num_classes)
            loss.backward()
            return loss
        
        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=400)
        optimizer.step(eval)

        best_temperature = self.temperature.item()
        while best_temperature<=0:
            self.temperature.data = torch.rand(1)
            optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=400)
            optimizer.step(eval)
            best_temperature = self.temperature.item()

        after_temperature_nll = nll_criterion(logits_in/best_temperature, labels_in).item()
        after_temperature_ece_out = ece_criterion_ood(logits_out/best_temperature, num_classes).item()
        after_temperature_ece = ece_criterion(logits_in/best_temperature, labels_in).item()
        print('Optimal temperature: %.3f' % best_temperature)
        print('After temperature - NLL: %.3f, ECE: %.3f, ECE_OUT: %.3f' % (after_temperature_nll, after_temperature_ece, after_temperature_ece_out))

        self.temp = best_temperature


    def forward(self, batched_inputs):
        images = batched_inputs['target_images'].to(self.device)
        indexs = batched_inputs['target_indexs'].to(self.device)
        
        features = self.backbone(images)

        logit = self.classifier(features)

        prior_q = F.softmax(self.target_logits[indexs] / self.temp, dim=-1)
        loss_distillation = -torch.sum(prior_q * torch.log(F.softmax(logit) + 1e-5), dim=1).mean()

        return loss_distillation
    
    def after_training(self):
        self.temperature.data = torch.tensor(self.temp)
        super().after_training()


##########################################
class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    
class _ECELossOOD(torch.nn.Module):
    """
    calculate the ECE loss for OOD (out-class) samples
    """
    def forward(self, logits, num_classes):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        ece = (confidences - 1/num_classes).mean()
        return ece

        
