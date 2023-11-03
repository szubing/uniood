import torch
from torch import nn
import torch.nn.functional as F

from datasets import dataset_classes
from methods.source_only import SourceOnly

# Official codes: https://github.com/VisionLearningGroup/DANCE
class Dance(SourceOnly):
    require_source = True
    require_target = True
    _hyparas = {'temperature': 0.05,
                'momentum': 0.0,
                'eta': 0.05,
                'entropy_margin': 0.5}
    score_mode = 'ENTROPY'
    
    def __init__(self, cfg) -> None:
        cfg.classifier_head = 'prototype'
        super().__init__(cfg)

        n_target_data = self.get_target_data_lengths(cfg)
        self.lemniscate = LinearAverage(self.feature_dim, n_target_data, self._hyparas['temperature'], self._hyparas['momentum']).to(self.device)


    def forward(self, batched_inputs):
        source_images = batched_inputs['source_images'].to(self.device)
        source_labels = batched_inputs['source_labels'].to(self.device)
        target_images = batched_inputs['target_images'].to(self.device)
        target_indexs = batched_inputs['target_indexs'].to(self.device)

        ## Weight normalizztion
        self.classifier.weight_norm()
        ## Source loss calculation
        feat = self.backbone(source_images)
        out_s = self.classifier(feat)
        loss_s = self.criterion(out_s, source_labels)

        feat_t = self.backbone(target_images)
        out_t = self.classifier(feat_t)
        feat_t = F.normalize(feat_t)
        ### update memory weights
        self.lemniscate.update_weight(feat_t, target_indexs)
        ### Calculate mini-batch x memory similarity
        feat_mat = self.lemniscate(feat_t, target_indexs)
        ### We do not use memory features present in mini-batch
        feat_mat[:, target_indexs] = -1 / self._hyparas['temperature']
        ### Calculate mini-batch x mini-batch similarity
        feat_mat2 = torch.matmul(feat_t,
                                 feat_t.t()) / self._hyparas['temperature']
        mask = torch.eye(feat_mat2.size(0),
                         feat_mat2.size(0)).bool().to(self.device)
        feat_mat2.masked_fill_(mask, -1 / self._hyparas['temperature'])
        loss_nc = self._hyparas['eta'] * self.get_entropy_from_logits(torch.cat([out_t, feat_mat, feat_mat2], 1)).mean()
        loss_ent = self._hyparas['eta'] * self.entropy_margin(out_t, -self.threshold, self._hyparas['entropy_margin'])
        loss = {'loss_nc': loss_nc, 'loss_source': loss_s, 'loss_entropy': loss_ent}
        return loss

    def get_target_data_lengths(self, cfg):
        data = dataset_classes[cfg.dataset](cfg.data_dir, cfg.source_domain, cfg.target_domain, cfg.n_share, cfg.n_source_private)
        return len(data.test)

    def entropy_margin(self, p, value, margin=0.2, weight=None):
        p = F.softmax(p)
        return -torch.mean(self.hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))

    def hinge(self, input, margin=0.2):
        return torch.clamp(input, min=margin)


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]))
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T
    def forward(self, x, y):
        out = torch.mm(x, self.memory.t())/self.T
        return out

    def update_weight(self, features, index):
        if not self.flag:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(0.0)
            weight_pos.add_(torch.mul(features.data, 1.0))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
            self.flag = 1
        else:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)#.cuda()


    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)