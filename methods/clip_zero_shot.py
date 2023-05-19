import torch
import clip
from torchvision.transforms import Normalize
from datasets import dataset_classes
from templates import get_templates, templates_types
from models import build_backbone, CLIP_MODELS
from methods.source_only import SourceOnly

CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

class ClipZeroShot(SourceOnly):
    """
    Implement zero shot classification by clip models
    """
    require_source = False
    require_target = False
    text_templetes = 'ensemble' # choises are ['classname', 'vanilla', 'hand_crafted', 'ensemble', 'template_mining']
    
    def __init__(self, cfg) -> None:
        cfg.fixed_backbone = True
        cfg.classifier_head = 'prototype'
        assert cfg.backbone in CLIP_MODELS, f'backbone must be in clip_available_models but got {cfg.backbone}'

        super().__init__(cfg)

        clip_model = build_backbone(cfg.backbone).to(self.device)
        
        self.classnames = self.get_classnames(cfg)
        self.templates = self.get_templates(cfg)
        print(f"{len(self.classnames)} classes, {len(self.templates)} templates")
        self.classifier.fc.weight.data, self.text_features, self.text_labels = text_information(clip_model, self.classnames, self.templates, self.device)

        self.logit_threshold =  self.get_logit_threshold(clip_model.visual)

        clip_model = None
        torch.cuda.empty_cache()

    def get_classnames(self, cfg):
        data = dataset_classes[cfg.dataset](cfg.data_dir, cfg.source_domain, cfg.target_domain, cfg.n_share, cfg.n_source_private)
        return data.classnames
    
    def get_templates(self, cfg):
        assert self.text_templetes in templates_types
        return get_templates(cfg.dataset, self.text_templetes)

    def forward(self, batched_inputs):
        pass

    def get_iid_scores(self, logits):
        max_logits, predict_labels = torch.max(logits, -1)
        return max_logits

    def predict_ood_indexs(self, logits):
        max_logits, _ = torch.max(logits, -1)
        ood_indexs = max_logits < self.logit_threshold
        return ood_indexs
    
    def get_logit_threshold(self, model, num_noise=100, sigma=3, noise_features=None):
        if noise_features is None:
            normalize = Normalize(mean=CLIP_PIXEL_MEAN, std=CLIP_PIXEL_STD)
            noise_imges = torch.rand(num_noise, 3, model.input_resolution, model.input_resolution)
            with torch.no_grad():
                noise_features = model(normalize(noise_imges.to(self.device)))
                self.noise_features = noise_features

        with torch.no_grad():
            noise_logits = self.classifier(noise_features)

        noise_max_logits, _ = noise_logits.max(-1)
        logit_threshold = noise_max_logits.max() + sigma*(noise_max_logits.max() - noise_max_logits.mean())
        print('logits_threshold', logit_threshold.item(), ' max:', noise_max_logits.max().item(), ' mean:', noise_max_logits.mean().item())

        return logit_threshold


##########################common functions to use
def text_information(clip_model, classnames, templates, device=0):
    with torch.no_grad():
        zeroshot_weights = []
        embeddings = []
        labels = []
        label = 0
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = clip_model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            embeddings.append(class_embeddings)
            labels += [label for i in range(len(class_embeddings))]
            label += 1
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.T, torch.cat(embeddings), torch.tensor(labels)



