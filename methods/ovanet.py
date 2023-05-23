import copy

import torch
import torch.nn.functional as F

from methods.source_only import SourceOnly
from models import build_head

# Official codes: https://github.com/VisionLearningGroup/OVANet
class OVANet(SourceOnly):
    require_source = True
    require_target = True
    _hyparas = {'loss_open_weight': 0.5,
                'loss_entropy_weight': 0.1}
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.open_classifier = build_head(cfg.classifier_head, self.feature_dim, 2*self.num_classes).to(self.device)

    def forward(self, batched_inputs):
        source_images = batched_inputs['source_images'].to(self.device)
        source_labels = batched_inputs['source_labels'].to(self.device)
        target_images = batched_inputs['target_images'].to(self.device)
        ## Source loss calculation
        feat = self.backbone(source_images)
        out_s = self.classifier(feat)
        out_open = self.open_classifier(feat)
        ## source classification loss
        loss_s = self.criterion(out_s, source_labels)
        ## open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1)
        open_loss_pos, open_loss_neg = self.ova_loss(out_open, source_labels)
        ## b x 2 x C
        loss_open = self._hyparas['loss_open_weight'] * (open_loss_pos + open_loss_neg)
        ## open set loss for target
        feat_t = self.backbone(target_images)
        out_open_t = self.open_classifier(feat_t)
        out_open_t = out_open_t.view(target_images.size(0), 2, -1)
        ent_open = self.open_entropy(out_open_t)
        ent_open_loss = self._hyparas['loss_entropy_weight'] * ent_open

        loss = {'loss_source': loss_s, 'loss_open': loss_open, 'loss_entropy': ent_open_loss}
        return loss
    
    def after_backward(self):
        if self.classifier_type == 'prototype':
            self.classifier.weight_norm()
            self.open_classifier.weight_norm()
    
    def predict(self, batched_inputs):
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
            logit_open = self.open_classifier(features)
            out_open = F.softmax(logit_open.view(logit.size(0), 2, -1),1)
            tmp_range = torch.range(0, logit.size(0)-1).long().to(self.device)
            pred_unk = out_open[tmp_range, 0, predict_labels]
            ood_indexs = torch.where(pred_unk.data > 0.5)[0]
            
            predict_labels[ood_indexs] = self.num_classes

            result_dict['features'] = features
            result_dict['predict_labels'] = predict_labels
            result_dict['logits'] = logit
            result_dict['predict_labels_without_ood'] = predict_labels_without_ood
            result_dict['iid_scores'] = 1.0 - pred_unk
        
        return result_dict
        
    
    def ova_loss(self, out_open, label):
        assert len(out_open.size()) == 3
        assert out_open.size(1) == 2

        out_open = F.softmax(out_open, 1)
        label_p = torch.zeros((out_open.size(0),
                            out_open.size(2))).long().to(self.device)
        label_range = torch.range(0, out_open.size(0) - 1).long()
        label_p[label_range, label] = 1
        label_n = 1 - label_p
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                        + 1e-8) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                    1e-8) * label_n, 1)[0])
        return open_loss_pos, open_loss_neg
    
    def open_entropy(self, out_open):
        assert len(out_open.size()) == 3
        assert out_open.size(1) == 2
        out_open = F.softmax(out_open, 1)
        ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
        return ent_open