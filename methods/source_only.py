import copy
import os

import torch
from torch import nn
import torch.nn.functional as F
from tools.utils import get_save_checkpoint_dir

from models import build_backbone, build_head, CLIP_MODELS, DINOv2_MODELS
from models.partial_model import get_partial_model

THRESHOLD_MODES = ('fixed', 'from_validation')
SCORE_MODES = ('MSP', 'MLS', 'ENTROPY', 'MarginP')

class SourceOnly(nn.Module):
    """
    Implement SO by ERM
    """
    require_source = True
    require_target = False
    threshold_mode = 'fixed' # 'fixed: set a fixed threshold; from_validataion: set the threshold based on the validation set'
    score_mode = 'ENTROPY'
    
    def __init__(self, cfg) -> None:
        super().__init__()
        """
        Build the torch models we want to optimize in here, 
            and note that the pretrained model should be named as 'self.backbone'.
        """
        assert self.threshold_mode in THRESHOLD_MODES
        assert self.score_mode in SCORE_MODES

        self.device = torch.device(cfg.device)
        self.num_classes = cfg.n_share + cfg.n_source_private

        self.backbone = build_backbone(cfg.backbone)

        if cfg.backbone in CLIP_MODELS:
            self.backbone = self.backbone.visual

        if cfg.fixed_backbone:
            assert cfg.ft_norm_only is False
            assert cfg.ft_last_layer is False
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if cfg.ft_norm_only:
            assert cfg.fixed_backbone is False
            assert cfg.ft_last_layer is False
            cfg.result_dir = os.path.join(cfg.result_dir, 'FT_NORM_ONLY')
            cfg.feature_dir = os.path.join(cfg.feature_dir, 'FT_NORM_ONLY')
            for key, value in self.backbone.named_parameters(recurse=True):
                if 'norm' in key:
                    value.requires_grad_(True)
                else:
                    value.requires_grad_(False)
        
        if cfg.ft_last_layer:
            assert cfg.fixed_backbone is False
            assert cfg.ft_norm_only is False
            cfg.result_dir = os.path.join(cfg.result_dir, 'FT_LAST_LAYER')
            cfg.feature_dir = os.path.join(cfg.feature_dir, 'FT_LAST_LAYER')
            fixed_model, self.partial_model = get_partial_model(self.backbone, layer_idx=1, name=cfg.backbone)
            for key, value in fixed_model.named_parameters(recurse=True):
                value.requires_grad_(False)
            for key, value in self.partial_model.named_parameters(recurse=True):
                value.requires_grad_(True)
            
            self.partial_model = self.partial_model.to(self.device)
        
        if cfg.backbone in DINOv2_MODELS:
            self.feature_dim = self.backbone.num_features
        else:
            self.feature_dim = self.backbone.output_dim

        self.classifier = build_head(cfg.classifier_head, self.feature_dim, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.backbone = self.backbone.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.criterion = self.criterion.to(self.device)

        if self.score_mode == 'ENTROPY':
            threshold = -torch.log(torch.tensor(self.num_classes)) / 2
        elif self.score_mode == 'MSP':
            threshold = 0.5
        elif self.score_mode == 'MLS':
            threshold = 0
        elif self.score_mode == 'MarginP':
            threshold = torch.sqrt(torch.tensor(self.num_classes-1))/self.num_classes
        self.set_threshold(threshold)

        self.fixed_backbone = cfg.fixed_backbone
        self.classifier_type = cfg.classifier_head
        self.fixed_BN = cfg.fixed_BN
        self.cfg = cfg

    def before_training(self, cfg=None):
        """
        This function initialize something that need to do before training the model.
        """
        self.train()
        if self.fixed_backbone:
            self.backbone.eval()
        if self.fixed_BN:
            for module in self.children():
                if isinstance(module, nn.BatchNorm2d):
                    module.train(False)

    def after_training(self):
        """
        This function implement something that need to do after training the model.
        """
        # save model
        if self.cfg.save_checkpoint:
            save_checkpoint_pth = self.get_save_checkpoint_dir(self.fixed_backbone)
            torch.save(self.state_dict(), save_checkpoint_pth)

    def get_save_checkpoint_dir(self, fixed_backbone):
        return get_save_checkpoint_dir(self.cfg.feature_dir, 
                                    f'{self.cfg.method}_{self.cfg.backbone}-{fixed_backbone}_{self.cfg.classifier_head}_{self.cfg.optimizer}_{self.cfg.batch_size}_{self.cfg.base_lr}_{self.cfg.fixed_BN}_{self.cfg.image_augmentation}_final-{self.cfg.max_iter}', 
                                    self.cfg.dataset, 
                                    self.cfg.source_domain, 
                                    self.cfg.target_domain, 
                                    self.cfg.n_share, 
                                    self.cfg.n_source_private, 
                                    self.cfg.seed)

    def before_forward(self):
        """
        This function implement something that need to do before each step of forward.
        """
        pass

    def forward(self, batched_inputs):
        """
        This function return a loss or a loss dict by receiving a dict of batched_inputs.
        1. if self.require_source == True, then the batched_inputs['source_images'] and batched_inputs['source_labels'] are not None;
        2. if self.require_target == True, then the batched_inputs['target_images'] and batched_inputs['target_indexs'] are not None;
        """
        source_images = batched_inputs['source_images'].to(self.device)
        source_labels = batched_inputs['source_labels'].to(self.device)
        
        features = self.backbone(source_images)
        logit = self.classifier(features)

        loss = self.criterion(logit, source_labels)

        return loss
    
    def after_backward(self):
        """
        This function implement something that need to do after each step of backward.
        """
        if self.classifier_type == 'prototype':
            self.classifier.weight_norm()

    def before_predict(self, cfg=None):
        self.eval()
        if isinstance(cfg, dict):
            # check wheather to set threshold based on the validation set
            if 'val_data_loader' in cfg.keys() and self.threshold_mode == 'from_validation':
                val_data_loader = cfg['val_data_loader']
                validation_iid_scores = []
                for batch_datas in val_data_loader:
                    batched_inputs = {'test_images': batch_datas['img']}
                    result_dict = self.predict(batched_inputs=batched_inputs)
                    validation_iid_scores.append(result_dict['iid_scores'].cpu().detach())
                
                val_scores = torch.cat(validation_iid_scores)
                threshold = val_scores.quantile(q=0.01)
                print('set threhold to be: ', threshold.item())
                self.set_threshold(threshold)
            
            if 'test_data_loader' in cfg.keys():
                self.test_dataloader = cfg['test_data_loader']
        
        # if self.score_mode == 'ENTROPY' and self.cfg.debug is not None:
        #     threshold = -torch.log(torch.tensor(self.num_classes)) / self.cfg.debug
        #     print('set threhold to be: ', threshold.item())
        #     self.set_threshold(threshold)
        #     self.cfg.method = self.cfg.method + str(self.cfg.debug)
        #     self.cfg.result_dir = self.cfg.result_dir + '_entropy'

        if self.cfg.eval_only:
            self.cfg.method = self.cfg.method + self.score_mode
            self.cfg.result_dir = self.cfg.result_dir + self.score_mode


    def predict(self, batched_inputs):
        """
        This function return the predict dict results after receiving a dict of batched_inputs.
        1. the return dict should includes following keys():
            e.g., result_dict = {'predict_labels': None,
                                 'predict_labels_without_ood': None,
                                 'features': None,
                                 'logits': None,
                                 'iid_scores': None}
        2. the batched_inputs includes the key of 'test_images', which is a tensor.
        """
        result_dict = {'predict_labels': None,
                       'predict_labels_without_ood': None,
                       'features': None,
                       'logits': None,
                       'iid_scores': None}
        with torch.no_grad():
            images = batched_inputs['test_images'].to(self.device)
            features = self.backbone(images)
            logit = self.classifier(features)
            max_logits, predict_labels = torch.max(logit, -1)
            predict_labels_without_ood = copy.deepcopy(predict_labels)
            # predict ood samples
            ood_indexs = self.predict_ood_indexs(logits=logit)
            predict_labels[ood_indexs] = self.num_classes

            result_dict['features'] = features
            result_dict['predict_labels'] = predict_labels
            result_dict['logits'] = logit
            result_dict['predict_labels_without_ood'] = predict_labels_without_ood
            result_dict['iid_scores'] = self.get_iid_scores(logit)
        
        return result_dict
    
    def get_iid_scores(self, logits):
        if self.score_mode == 'ENTROPY':
            return -self.get_entropy_from_logits(logits)
        elif self.score_mode == 'MLS':
            max_logits, _ = torch.max(logits, -1)
            return max_logits
        elif self.score_mode == 'MSP':
            max_probs, _ = torch.max(F.softmax(logits, dim=-1), -1)
            return max_probs
        elif self.score_mode == 'MarginP':
            probs = F.softmax(logits, dim=-1)
            sort_probs = probs.sort(-1)[0]
            return sort_probs[:,-1] - sort_probs[:,-2]
        else:
            raise ValueError

    def get_entropy_from_logits(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy_values = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
        return entropy_values
    
    def predict_ood_indexs(self, logits):
        iid_scores = self.get_iid_scores(logits)
        ood_indexs = iid_scores <= self.threshold
        return ood_indexs
    
    def set_threshold(self, threshold):
        if isinstance(threshold, torch.Tensor):
            self.threshold = threshold.to(self.device)
        elif isinstance(threshold, int) or isinstance(threshold, float):
            self.threshold = threshold
        else:
            ValueError('threshold type not support')
            



