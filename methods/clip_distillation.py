import copy
import random
import torch
import torch.nn.functional as F

from models import CLIP_MODELS
from methods.source_only import SourceOnly
from tools.utils import get_save_logits_dir, get_save_checkpoint_dir

class ClipDistill(SourceOnly):
    require_source = False
    require_target = True
    _hyparas = {'source_black_box': 
                {'office31': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'officehome': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'visda': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                 'domainnet': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}'},

                 'temperature': 0.3,
                }
    # hyperas_analysis = True

    def __init__(self, cfg) -> None:
        self.in_clip = cfg.backbone in CLIP_MODELS
        if self.in_clip:
            cfg.fixed_backbone = True
            black_box_backbone = cfg.backbone.replace('/', '')
            black_box_method = 'ClipZeroShot'
            seed = 1
        else:
            raise NotImplementedError()

        cfg.classifier_head = 'prototype'

        if cfg.debug is not None: # hyperas_analysis = True
            cfg.method = f'debug{cfg.debug}' # for debug use
            self.temp = cfg.debug
        else:
            self.temp = self._hyparas['temperature']
        
        super().__init__(cfg)

        file_path = get_save_logits_dir(cfg.feature_dir, 
                                        self._hyparas['source_black_box'][cfg.dataset].format(black_box_method, black_box_backbone, cfg.max_iter), 
                                        cfg.dataset, 
                                        cfg.source_domain, 
                                        cfg.target_domain, 
                                        cfg.n_share, 
                                        cfg.n_source_private, 
                                        seed)
        
        self.target_logits = torch.load(file_path).to(self.device)
        self.logits, self.pseudo_labels = torch.max(self.target_logits, dim=-1)

        self.file_path_checkpoint = get_save_checkpoint_dir(cfg.feature_dir, 
                                                       self._hyparas['source_black_box'][cfg.dataset].format(black_box_method, black_box_backbone, cfg.max_iter), 
                                                       cfg.dataset, 
                                                       cfg.source_domain, 
                                                       cfg.target_domain, 
                                                       cfg.n_share, 
                                                       cfg.n_source_private, 
                                                       seed)
        
        ## for test only
        # self.file_path_checkpoint = get_save_checkpoint_dir(cfg.feature_dir, 
        #                             f'UB2DA_{cfg.backbone}-{cfg.fixed_backbone}_{cfg.classifier_head}_{cfg.optimizer}_{cfg.batch_size}_{cfg.base_lr}_{cfg.fixed_BN}_{cfg.image_augmentation}_final-{cfg.max_iter}', 
        #                             cfg.dataset, 
        #                             cfg.source_domain, 
        #                             cfg.target_domain, 
        #                             cfg.n_share, 
        #                             cfg.n_source_private, 
        #                             cfg.seed)
       
        self.alpha = None #
    
    def before_training(self, cfg=None):
        super().before_training(cfg)
        
        if self.in_clip:
            state_dict = torch.load(self.file_path_checkpoint)
            self.load_state_dict(state_dict, strict=False)
            self.prototypes = copy.deepcopy(self.classifier.fc.weight.data)
    

    def forward(self, batched_inputs):
        images = batched_inputs['target_images'].to(self.device)
        indexs = batched_inputs['target_indexs'].to(self.device)
        
        features = self.backbone(images)

        logit = self.classifier(features)

        prior_q = F.softmax(self.target_logits[indexs] / self.temp, dim=-1)
        loss_distillation = -torch.sum(prior_q * torch.log(F.softmax(logit) + 1e-5), dim=1).mean()

        return loss_distillation
    
    def after_training(self):
        super().after_training()
        if self.alpha is not None and self.in_clip:
            self.classifier.fc.weight.data = (1-self.alpha)*self.prototypes.data + self.alpha*self.classifier.fc.weight.data
        
