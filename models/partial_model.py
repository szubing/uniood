import torch
from torch import nn

from models import CLIP_MODELS

class PartialViT(nn.Module):
    def __init__(self, conv1=None,
                       class_embedding=None,
                       positional_embedding=None,
                       ln_pre=None,
                       transformer_encoder=None,
                       ln_post=None,
                       proj=None,
                 mode='feature_extractor'):
        super().__init__()
        assert mode in ['feature_extractor', 'partial_model']
        self.conv1 = conv1
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.ln_pre = ln_pre
        self.transformer_encoder = transformer_encoder
        self.ln_post = ln_post
        self.proj = proj
        if mode == 'partial_model':
            if self.conv1 is not None:
                assert self.ln_pre is not None
            if self.ln_pre is not None:
                assert self.transformer_encoder is not None
            if self.transformer_encoder is not None:
                assert self.ln_post is not None
            if self.ln_post is not None:
                assert self.proj is not None
        elif mode == 'feature_extractor':
            if self.proj is not None:
                assert self.ln_post is not None
            if self.ln_post is not None:
                assert self.transformer_encoder is not None
            if self.transformer_encoder is not None:
                assert self.ln_pre is not None
            if self.ln_pre is not None:
                assert self.conv1 is not None
    
    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        if self.class_embedding is not None:
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        if self.positional_embedding is not None:
            x = x + self.positional_embedding.to(x.dtype)
        
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        
        if self.transformer_encoder is not None:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        
        if self.ln_post is not None:
            x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj

        return x


def get_split_vit_clip(model, layer_idx=0):
    # contains feature_extractor and partial_model
    conv1 = model.conv1
    class_embedding = model.class_embedding
    positional_embedding = model.positional_embedding
    ln_pre = model.ln_pre
    transformer = model.transformer
    ln_post = model.ln_post
    proj = model.proj

    if layer_idx == -1:
        # finetune all layers
        feature_extractor = PartialViT(mode='feature_extractor')
        partial_model = PartialViT(conv1=conv1,
                                   class_embedding=class_embedding,
                                   positional_embedding=positional_embedding,
                                   ln_pre=ln_pre,
                                   transformer_encoder=transformer,
                                   ln_post=ln_post,
                                   proj=proj,
                                   mode='partial_model')
    elif layer_idx == 0:
        # finetune no layers
        feature_extractor = PartialViT(conv1=conv1,
                                       class_embedding=class_embedding,
                                       positional_embedding=positional_embedding,
                                       ln_pre=ln_pre,
                                       transformer_encoder=transformer,
                                       ln_post=ln_post,
                                       proj=proj,
                                       mode='feature_extractor')
        partial_model = PartialViT(mode='partial_model')
    else:
        # finetune some layers
        transformer_encoder = transformer.resblocks[:-layer_idx]
        partial_transformer = transformer.resblocks[-layer_idx:]
        feature_extractor = PartialViT(conv1=conv1,
                                       class_embedding=class_embedding,
                                       positional_embedding=positional_embedding,
                                       ln_pre=ln_pre,
                                       transformer_encoder=transformer_encoder,
                                       mode='feature_extractor')
        partial_model = PartialViT(transformer_encoder=partial_transformer,
                                   ln_post=ln_post,
                                   proj=proj,
                                   mode='partial_model')
    feature_extractor.eval()
    partial_model.train()
    return feature_extractor, partial_model


def get_partial_model(model, layer_idx, name):
    if name in CLIP_MODELS:
        return get_split_vit_clip(model, layer_idx=layer_idx)
    else:
        raise ValueError(name + ' not support for building partial model')