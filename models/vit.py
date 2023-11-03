# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Modified by Bin Deng

# from turtle import forward
import torch
import torch.nn as nn
from functools import partial

# import sys
# sys.path.append('/home/lab-deng.bin/coding/JUSTforLEAENING/pytorch-image-models')
from timm.models.vision_transformer import VisionTransformer


class deit_base(VisionTransformer):
    def __init__(self, pretrained=True):
        super().__init__(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.head = None

        self.output_dim = 768

        if pretrained:
            self._my_load_from_state_dict()

    def get_feature_dim(self):
        return self.output_dim

    def _my_load_from_state_dict(self, url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth"):
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url,
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint:
            param_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            param_dict = checkpoint['state_dict']
        else:
            param_dict = checkpoint
        
        load_flag = True
        for k, v in param_dict.items():
            if 'dist' in k:
                continue
            if k == 'head.weight':
                continue
            if k == 'head.bias':
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at without distillation pos
                v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
                load_flag = False

        if load_flag:
            print(f'load_state_dict from {url} successful')
        else:
            raise Exception(f'load_state_dict from {url} fail')

    def forward(self, x):
        feat = super().forward_features(x)
        if feat.dim() == 3:
            feat = feat[:, 0]
        return feat


class vit_base(deit_base):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=False)

        if pretrained:
            self._my_load_from_state_dict(url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth")


class vit_base_dino(deit_base):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=False)

        if pretrained:
            self._my_load_from_state_dict(url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth")


################################################
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)