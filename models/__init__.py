import clip

from models.resnet import ResBase
from models.classifier import CLS, ProtoCLS, Projection, ProtoNormCLS


CLIP_MODELS = clip.available_models()
DINOv2_MODELS = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']
backbone_names = ['resnet50'] + CLIP_MODELS + DINOv2_MODELS
head_names = ['linear', 'mlp', 'prototype', 'protonorm']


def build_backbone(name):
    """
    build the backbone for feature exatraction
    """
    if name == 'resnet50':
        return ResBase(option=name)
    elif name in CLIP_MODELS:
        model, _ = clip.load(name)
    elif name in DINOv2_MODELS:
        import torch
        model = torch.hub.load('facebookresearch/dinov2', name)
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