import copy
import random
import numpy as np
import os
import torch
import torch.nn.functional as F

from models import CLIP_MODELS
from methods.source_only import SourceOnly
from tools.utils import get_save_logits_dir, get_save_checkpoint_dir

class Distill(SourceOnly):
    require_source = True
    require_target = True
    _hyparas = {'source_black_box': 
                {'office31': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'officehome': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'visda': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'domainnet': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}'},

                 'temperature': 0.3,
                }
    # threshold_mode = 'from_validation'
    # hyperas_analysis = True

    def __init__(self, cfg) -> None:

        cfg.image_augmentation = 'twoCrops'

        black_box_backbone = 'ViT-L14@336px'
        black_box_method = 'ClipZeroShot'
        seed = 1

        cfg.classifier_head = 'prototype'

        if cfg.debug is not None: # hyperas_analysis = True
            cfg.method = f'{cfg.method}-debug{cfg.debug}' # for debug use
            self.temp = cfg.debug
        else:
            self.temp = self._hyparas['temperature']
        
        super().__init__(cfg)

        self.bce = torch.nn.BCELoss()

        file_path = get_save_logits_dir(cfg.feature_dir, 
                                        self._hyparas['source_black_box'][cfg.dataset].format(black_box_method, black_box_backbone, cfg.max_iter), 
                                        cfg.dataset, 
                                        cfg.source_domain, 
                                        cfg.target_domain, 
                                        cfg.n_share, 
                                        cfg.n_source_private, 
                                        seed)
        
        self.target_logits = torch.load(file_path).to(self.device)
        # self.logits, self.pseudo_labels = torch.max(self.target_logits, dim=-1)

        # self.file_path_checkpoint = get_save_checkpoint_dir(cfg.feature_dir, 
        #                                                self._hyparas['source_black_box'][cfg.dataset].format(black_box_method, black_box_backbone, cfg.max_iter), 
        #                                                cfg.dataset, 
        #                                                cfg.source_domain, 
        #                                                cfg.target_domain, 
        #                                                cfg.n_share, 
        #                                                cfg.n_source_private, 
        #                                                seed)
                
        distill_from = 'distill_from-' + black_box_backbone + '-' + black_box_method + '-' + str(seed)
        cfg.result_dir = os.path.join(cfg.result_dir, distill_from)
        cfg.feature_dir = os.path.join(cfg.feature_dir, distill_from)
        

    def forward(self, batched_inputs):
        source_images_1, source_images_2 = batched_inputs['source_images']
        source_labels = batched_inputs['source_labels'].to(self.device)
        target_images_1, target_images_2 = batched_inputs['target_images']
        indexs = batched_inputs['target_indexs'].to(self.device)

        x = torch.cat([source_images_1, target_images_1], 0)
        x2 = torch.cat([source_images_2, target_images_2], 0)
        labeled_len = len(source_labels)

        x, x2= x.to(self.device), x2.to(self.device)
        feat = self.backbone(x)
        feat2 = self.backbone(x2)

        output = self.classifier(feat)
        output2 = self.classifier(feat2)

        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)
        
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())

        pos_pairs = []
        target_np = source_labels.cpu().numpy()
        
        # label part
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))

        # unlabel part
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)
        
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(labeled_len+len(indexs), 1, -1), pos_prob.view(labeled_len+len(indexs), -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = self.bce(pos_sim, ones)
        ce_loss = self.criterion(output[:labeled_len], source_labels)

        prior_q = F.softmax(self.target_logits[indexs] / self.temp, dim=-1)
        loss_distillation = -torch.sum(prior_q * torch.log(F.softmax(output[labeled_len:]) + 1e-5), dim=1).mean()

        loss = {'entropy_loss': ce_loss, 'pair_loss': bce_loss, 'distill_loss': loss_distillation}
        
        return loss

        
