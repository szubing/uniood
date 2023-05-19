import copy

import torch
from torch import nn
import torch.nn.functional as F
from tools.utils import get_save_checkpoint_dir

from models import build_backbone, build_head, CLIP_MODELS, DINOv2_MODELS

class SourceOnly(nn.Module):
    """
    Implement SO by ERM
    """
    require_source = True
    require_target = False
    
    def __init__(self, cfg) -> None:
        super().__init__()
        """
        Build the torch models we want to optimize in here, 
            and note that the pretrained model should be named as 'self.backbone'.
        """

        self.device = torch.device(cfg.device)
        self.num_classes = cfg.n_share + cfg.n_source_private

        self.backbone = build_backbone(cfg.backbone)

        if cfg.backbone in CLIP_MODELS:
            self.backbone = self.backbone.visual

        if cfg.fixed_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        
        if cfg.backbone in DINOv2_MODELS:
            self.feature_dim = self.backbone.num_features
        else:
            self.feature_dim = self.backbone.output_dim

        self.classifier = build_head(cfg.classifier_head, self.feature_dim, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.backbone = self.backbone.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.criterion = self.criterion.to(self.device)

        self.entropy_threshold = torch.log(torch.tensor(self.num_classes)) / 2
        self.entropy_threshold = self.entropy_threshold.to(self.device)

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

    def predict(self, batched_inputs):
        """
        This function return the predict dict results after receiving a dict of batched_inputs.
        1. the return dict should includes folloing keys():
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
        return -self.get_entropy_from_logits(logits)

    def get_entropy_from_logits(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy_values = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
        return entropy_values
    
    def predict_ood_indexs(self, logits):
        entropy_values = self.get_entropy_from_logits(logits)
        ood_indexs = entropy_values > self.entropy_threshold
        return ood_indexs

            



