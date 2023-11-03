import os
import torch
from torch.utils.data import DataLoader
import clip

from datasets import dataset_classes
from datasets.utils import TensorDataset
from templates import get_templates, templates_types
from models import build_backbone, CLIP_MODELS
from methods.clip_zero_shot import ClipZeroShot

# Official codes: https://github.com/linzhiqiu/cross_modal_adaptation
class ClipCrossModel(ClipZeroShot):
    """
    Implement zero shot classification by clip models
    """
    require_source = True
    require_target = False
    text_templetes = 'ensemble' # choises are ['classname', 'vanilla', 'hand_crafted', 'ensemble', 'template_mining']
    # threshold_mode = 'from_validation'
    
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        text_dataset =  TensorDataset(self.text_features, self.text_labels)
        self.text_data_loader = DataLoader(text_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            drop_last=False,)
        self.text_loader_iter = iter(self.text_data_loader)

    def forward(self, batched_inputs):
        if self.require_source:
            source_images = batched_inputs['source_images'].to(self.device)
            source_labels = batched_inputs['source_labels'].to(self.device)
            features_image = self.backbone(source_images)
            logit_image = self.classifier(features_image)
            loss_image = self.criterion(logit_image, source_labels)
        else:
            loss_image = torch.tensor(0).to(self.device)

        try:
            text_features, text_labels = next(self.text_loader_iter)
        except StopIteration:
            self.text_loader_iter = iter(self.text_data_loader)
            text_features, text_labels = next(self.text_loader_iter)

        logit_text = self.classifier(text_features.to(self.device))
        loss_text = self.criterion(logit_text, text_labels.to(self.device))

        loss = {'loss_image': loss_image, 'loss_text': loss_text}

        return loss

    # def get_logit_threshold(self, model, num_noise=100, sigma=3, noise_features=None):
    #     if not self.require_source:
    #         return super().get_logit_threshold(model, num_noise, sigma, noise_features)
    #     else:
    #         return None

    # def predict_ood_indexs(self, logits):
    #     if self.require_source:
    #         entropy_values = self.get_entropy_from_logits(logits)
    #         ood_indexs = entropy_values > self.entropy_threshold
    #     else:
    #         ood_indexs = super().predict_ood_indexs(logits)
    #     return ood_indexs
    
    # def after_training(self):
    #     super().after_training()
    #     if not self.require_source:
    #         self.logit_threshold = self.get_logit_threshold(None, noise_features=self.noise_features)

    # def get_iid_scores(self, logits):
    #     return -self.get_entropy_from_logits(logits)
            
    # def before_training(self, cfg=None):
    #     super().before_training(cfg)
    #     save_checkpoint_pth = self.get_save_checkpoint_dir(fixed_backbone=True)
    #     if os.path.exists(save_checkpoint_pth) and not self.fixed_backbone:
    #         state_dict = torch.load(save_checkpoint_pth)
    #         self.load_state_dict(state_dict, strict=False)
    #         print(f'initialize the classfier from the linear-prob classifier from {save_checkpoint_pth}')


            



