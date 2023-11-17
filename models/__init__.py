import clip

from models.resnet import ResBase
from models.classifier import CLS, ProtoCLS, Projection, ProtoNormCLS
from models.vit import vit_base, vit_base_dino, deit_base


CLIP_MODELS = clip.available_models()
DINOv2_MODELS = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']
backbone_names = ['resnet18', 'resnet50'] + CLIP_MODELS + DINOv2_MODELS + ['vit_base', 'vit_base_dino', 'deit_base']
head_names = ['linear', 'mlp', 'prototype', 'protonorm']


def build_backbone(name):
    """
    build the backbone for feature exatraction
    """
    if 'resnet' in name:
        return ResBase(option=name)
    elif name in CLIP_MODELS:
        model, _ = clip.load(name)
    elif name in DINOv2_MODELS:
        import torch
        model = torch.hub.load('facebookresearch/dinov2', name)
        # model = torch.hub.load('/data1/deng.bin/coding/JUSTforLearning/dinov2', name, source='local')
    elif name == 'vit_base':
        model = vit_base(pretrained=False)
    elif name == 'vit_base_dino':
        model = vit_base_dino()
    elif name == 'deit_base':
        model = deit_base()
    else:
        raise RuntimeError(f"Model {name} not found; available models = {backbone_names}")
    
    return model.float()


def build_head(name, in_dim, out_dim, hidden_dim=2048, temp=0.05):
    if name == 'linear':
        return CLS(in_dim, out_dim)
    elif name == 'mlp':
        return Projection(in_dim, feat_dim=out_dim, hidden_mlp=hidden_dim)
    elif name == 'prototype':
        return ProtoCLS(in_dim, out_dim, temp=temp)
    elif name == 'protonorm':
        return ProtoNormCLS(in_dim, out_dim, temp=temp)
    else:
        raise RuntimeError(f"Model {name} not found; available models = {head_names}")