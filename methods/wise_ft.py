import copy
import torch

from models import CLIP_MODELS
from methods.source_only import SourceOnly
from tools.utils import get_save_checkpoint_dir

# Official codes: https://github.com/mlfoundations/wise-ft
class WiSE_FT(SourceOnly):
    require_source = False
    require_target = False
    _hyparas = {'checkpoint': {'office31': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                        'officehome': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                        'visda': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}',
                        'domainnet': '{}_{}-True_prototype_sgd_32_0.01_False_none_final-{}'},
                 'alpha': 0.5}
    # threshold_mode = 'from_validation'

    def __init__(self, cfg) -> None:
        self.in_clip = cfg.backbone in CLIP_MODELS
        assert self.in_clip
        cfg.fixed_backbone = True
        backbone = cfg.backbone.replace('/', '')
        cfg.classifier_head = 'prototype'
        
        super().__init__(cfg)


        self.file_path_checkpoint_so = get_save_checkpoint_dir(cfg.feature_dir, 
                                                       self._hyparas['checkpoint'][cfg.dataset].format('SO', backbone, cfg.max_iter), 
                                                       cfg.dataset, 
                                                       cfg.source_domain, 
                                                       cfg.target_domain, 
                                                       cfg.n_share, 
                                                       cfg.n_source_private, 
                                                       cfg.seed)
        
        self.file_path_checkpoint_zeroshot = get_save_checkpoint_dir(cfg.feature_dir, 
                                                       self._hyparas['checkpoint'][cfg.dataset].format('ClipZeroShot', backbone, cfg.max_iter), 
                                                       cfg.dataset, 
                                                       cfg.source_domain, 
                                                       cfg.target_domain, 
                                                       cfg.n_share, 
                                                       cfg.n_source_private, 
                                                       1)
        
       
        self.alpha = self._hyparas['alpha']
    
    def before_training(self, cfg=None):
        super().before_training(cfg)
        
        state_dict_so = torch.load(self.file_path_checkpoint_so)
        self.load_state_dict(state_dict_so, strict=False)
        self.prototypes_so = copy.deepcopy(self.classifier.fc.weight.data)

        state_dict_zeroshot = torch.load(self.file_path_checkpoint_zeroshot)
        self.load_state_dict(state_dict_zeroshot, strict=False)
        self.prototypes_zeroshot = copy.deepcopy(self.classifier.fc.weight.data)

        self.classifier.fc.weight.data = self.alpha * self.prototypes_so + (1 - self.alpha) * self.prototypes_zeroshot
    

    def forward(self, batched_inputs):
        pass
        
