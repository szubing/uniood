import torch
from torch import nn
import torch.nn.functional as F
import ot

from methods.source_only import SourceOnly
from models import build_head

# Official codes: https://github.com/changwxx/UniOT-for-UniDA
class UniOT(SourceOnly):
    require_source = True
    require_target = True
    _hyparas = {'num_cluster': {'office31': 50, 'officehome': 150, 'visda': 500, 'domainnet': 1000},
                'memory_length':{'office31': 2000, 'officehome': 4000, 'visda': 5000, 'domainnet': 5000},
                'mu': {'office31': 0.7, 'officehome': 1.0, 'visda': 0.7, 'domainnet': 1.0},
                'feat_dim': 256,
                'temperature': 0.1,
                'gamma': 0.7,
                'lam': 0.1}
    def __init__(self, cfg) -> None:
        cfg.classifier_head = 'prototype'
        assert cfg.max_iter > 100
        super().__init__(cfg)
        
        self.K = self._hyparas['num_cluster'][cfg.dataset]
        self.mu = self._hyparas['mu'][cfg.dataset]

        self.projection_head = build_head('mlp', in_dim=self.feature_dim, out_dim=self._hyparas['feat_dim'], hidden_dim=2048).to(self.device)
        self.classifier = build_head('prototype', self._hyparas['feat_dim'], self.num_classes, temp=self._hyparas['temperature']).to(self.device)
        self.cluster_head = build_head('prototype', in_dim=self._hyparas['feat_dim'], out_dim=self.K, temp=self._hyparas['temperature']).to(self.device)
        
        self.n_batch = int(self._hyparas['memory_length'][cfg.dataset]/cfg.batch_size)
        self.memqueue = MemoryQueue(self._hyparas['feat_dim'],
                                    cfg.batch_size, 
                                    self.n_batch, 
                                    T=self._hyparas['temperature']).to(self.device)
        

        self.predict_features = []
        self.global_step = 0
        self.batch_feat_update = None
        self.batch_index_updata = None

    def before_training(self, cfg=None):
        super().before_training()
        if isinstance(cfg, dict): # cfg is not None means that this process is done in the initializing time
            self.global_step = 0
            self.target_dataloader, self.test_dataloader = cfg['target_data_loader'], cfg['test_data_loader']
            with torch.no_grad():
                cnt_i = 0
                while cnt_i < self.n_batch:
                    for batch_datas in self.target_dataloader:
                        im_target = batch_datas['img'].to(self.device)
                        id_target = batch_datas['idx'].to(self.device)
                        feature_ex = self.backbone(im_target)
                        before_lincls_feat = self.projection_head(feature_ex)
                        if self.memqueue.update_queue(F.normalize(before_lincls_feat), id_target):
                            cnt_i += 1
                        if cnt_i == self.n_batch:
                            break

    def forward(self, batched_inputs):
        source_images = batched_inputs['source_images'].to(self.device)
        source_labels = batched_inputs['source_labels'].to(self.device)
        target_images = batched_inputs['target_images'].to(self.device)
        target_indexs = batched_inputs['target_indexs'].to(self.device)

        feature_ex_s = self.backbone.forward(source_images)
        feature_ex_t = self.backbone.forward(target_images)

        before_lincls_feat_s  = self.projection_head(feature_ex_s)
        before_lincls_feat_t  = self.projection_head(feature_ex_t)
        after_lincls_s = self.classifier(before_lincls_feat_s)
        after_lincls_t = self.classifier(before_lincls_feat_t)

        norm_feat_t = F.normalize(before_lincls_feat_t)

        after_cluhead_t = self.cluster_head(before_lincls_feat_t)

        # self.memqueue.update_queue(norm_feat_t, target_indexs)
        self.batch_feat_update = norm_feat_t.data
        self.batch_index_updata = target_indexs.data
        # =====Source Supervision=====
        loss_cls = self.criterion(after_lincls_s, source_labels)

        # =====Private Class Discovery=====
        minibatch_size = norm_feat_t.size(0)

        # obtain nearest neighbor from memory queue and current mini-batch
        feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / self._hyparas['temperature']
        mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(self.device)
        feat_mat2.masked_fill_(mask, -1/self._hyparas['temperature'])

        nb_value_tt, nb_feat_tt = self.memqueue.get_nearest_neighbor(norm_feat_t, target_indexs)
        neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1,1), feat_mat2], 1)
        values, indices = torch.max(neighbor_candidate_sim, 1)
        neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).to(self.device)
        for i in range(minibatch_size):
            neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1,-1), norm_feat_t], 0)
            neighbor_norm_feat[i,:] = neighbor_candidate_feat[indices[i],:]
            
        neighbor_output = self.cluster_head(neighbor_norm_feat)
        
        # fill input features with memory queue
        fill_size_ot = self.K
        mqfill_feat_t = self.memqueue.random_sample(fill_size_ot)
        mqfill_output_t = self.cluster_head(mqfill_feat_t)

        # OT process
        # mini-batch feat (anchor) | neighbor feat | filled feat (sampled from memory queue)
        S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
        S_tt *= self._hyparas['temperature']
        Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
        Q_tt_tilde = Q_tt * Q_tt.size(0)
        anchor_Q = Q_tt_tilde[:minibatch_size, :]
        neighbor_Q = Q_tt_tilde[minibatch_size:2*minibatch_size, :]

        # compute loss_PCD
        loss_local = torch.tensor(0).float().to(self.device)
        for i in range(minibatch_size):
            sub_loss_local = 0
            sub_loss_local += -torch.sum(neighbor_Q[i,:] * F.log_softmax(after_cluhead_t[i,:], dim=-1))
            sub_loss_local += -torch.sum(anchor_Q[i,:] * F.log_softmax(neighbor_output[i,:], dim=-1))
            sub_loss_local /= 2
            loss_local += sub_loss_local
        loss_local /= minibatch_size
        loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
        loss_PCD = self._hyparas['lam'] * (loss_global + loss_local) / 2

        # =====Common Class Detection=====
        if self.global_step > 100:  
            self.beta = ot.unif(self.num_classes)
            # fill input features with memory queue
            fill_size_uot = self.memqueue.queue_size
            mqfill_feat_t = self.memqueue.random_sample(fill_size_uot)
            ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
            
            # Adaptive filling
            newsim, fake_size = self.adaptive_filling(ubot_feature_t, self.classifier.fc.weight, self._hyparas['gamma'], self.beta, fill_size_uot)
        
            # UOT-based CCD
            high_conf_label_id, high_conf_label, _, new_beta, _, _ = self.ubot_CCD(newsim, self.beta, fake_size=fake_size, 
                                                                    fill_size=fill_size_uot, mode='minibatch')
            # adaptive update for marginal probability vector
            self.beta = self.mu*self.beta + (1-self.mu)*new_beta

            # fix the bug raised in https://github.com/changwxx/UniOT-for-UniDA/issues/1
            # Due to mini-batch sampling, current mini-batch samples might be all target-private. 
            # (especially when target-private samples dominate target domain, e.g. OfficeHome)
            if high_conf_label_id.size(0) > 0:
                loss_CCD = self._hyparas['lam'] * self.criterion(after_lincls_t[high_conf_label_id,:], high_conf_label[high_conf_label_id])
            else:
                loss_CCD = torch.tensor(0).float().to(self.device)
        else:
            loss_CCD = torch.tensor(0).float().to(self.device)
        
        loss = {'loss_source': loss_cls, 'loss_PCD': loss_PCD, 'loss_CCD': loss_CCD}
        return loss
    
    def after_backward(self):
        self.classifier.weight_norm() # very important for proto-classifier
        self.cluster_head.weight_norm() # very important for proto-classifier
        self.memqueue.update_queue(self.batch_feat_update, self.batch_index_updata)
        self.global_step += 1

    def predict(self, batched_inputs):
        result_dict = {'predict_labels': None,
                       'predict_labels_without_ood': None,
                       'features': None,
                       'logits': None,
                       'iid_scores': None}
        with torch.no_grad():
            images = batched_inputs['test_images'].to(self.device)
            features = self.backbone(images)
            features = self.projection_head(features)
            logit = self.classifier(features)

            max_logits, _ = torch.max(logit, -1)

            self.predict_features.append(features)
            if len(torch.cat(self.predict_features)) == len(self.test_dataloader.dataset):
                predict_labels, predict_labels_without_ood, iid_scores = self.predict_only_once()
            else:
                predict_labels = None
                predict_labels_without_ood = None
                iid_scores = None

            result_dict['features'] = features
            result_dict['predict_labels'] = predict_labels
            result_dict['logits'] = logit
            result_dict['predict_labels_without_ood'] = predict_labels_without_ood
            result_dict['iid_scores'] = iid_scores
        
        return result_dict
    
    def predict_only_once(self):
        # predict ood samples
        # Unbalanced OT
        source_prototype = self.classifier.fc.weight
        # Adaptive filling 
        newsim, fake_size = self.adaptive_filling(F.normalize(torch.cat(self.predict_features)), 
                                            source_prototype, self._hyparas['gamma'], self.beta, 0, stopThr=1e-6)

        # obtain predict label
        _, __, predict_labels, ___, predict_labels_without_ood, iid_scores = self.ubot_CCD(newsim, self.beta, fake_size=fake_size, 
                                                                                fill_size=0, mode='minibatch', stopThr=1e-6)
        self.predict_features = []
        return predict_labels, predict_labels_without_ood, iid_scores
    
    def adaptive_filling(self, ubot_feature_t, source_prototype, gamma, beta, fill_size, stopThr=1e-4):
        sim = torch.matmul(ubot_feature_t, source_prototype.t())
        max_sim, _ = torch.max(sim,1)
        pos_id = torch.nonzero(max_sim > gamma).reshape(-1)
        pos_rate = pos_id.size(0)/max_sim.size(0)
        pos_num = pos_id.size(0)
        neg_num = max_sim.size(0) - pos_num
        if pos_rate <= 0.5:
            # positive filling
            fake_size = neg_num - pos_num
            if fake_size > 0:
                # do 1st OT find top confident target samples
                high_conf_label_id, _, __, ___,____,_____= self.ubot_CCD(sim, beta, fake_size=0, fill_size=fill_size, 
                                                        mode='all', stopThr=stopThr)
                if high_conf_label_id.size(0) > 0:
                    select_id = torch.randint(0, high_conf_label_id.size(0), (fake_size,)).to(self.device)
                    fill_pos = sim[high_conf_label_id[select_id],:] 
                    newsim = torch.cat([fill_pos, sim], 0)
                else:
                    fake_size = 0
                    newsim = sim
            else:
                newsim = sim
        else:
            # negative filling
            fake_size = pos_num - neg_num
            if fake_size > 0:
                farthest_sproto_id = torch.argmin(sim, 1)
                fake_private = 0.5 * ubot_feature_t + 0.5 * source_prototype.data[farthest_sproto_id,:]
                fake_private = F.normalize(fake_private)
                select_id = torch.randint(0, fake_private.size(0), (fake_size,))
                fill_neg = fake_private[select_id,:]
                fake_sim = torch.matmul(fill_neg, source_prototype.t())
                newsim = torch.cat([fake_sim, sim], 0)
            else:
                newsim = sim
        
        return newsim, fake_size


    def ubot_CCD(self, sim, beta, fake_size=0, fill_size=0, mode='minibatch', stopThr=1e-4):
        # fake_size (Adaptive filling) + fill_size (memory queue filling) + mini-batch size
        M = -sim                         
        alpha = ot.unif(sim.size(0))
        
        Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(alpha, beta, M.detach().cpu().numpy(), 
                                                        reg=0.01, reg_m=0.5, stopThr=stopThr) 
        Q_st = torch.from_numpy(Q_st).float().to(self.device)

        # make sum equals to 1
        sum_pi = torch.sum(Q_st)
        Q_st_bar = Q_st/sum_pi
        
        # highly confident target samples selected by statistics mean
        if mode == 'minibatch':
            Q_anchor = Q_st_bar[fake_size+fill_size:, :]
        if mode == 'all':
            Q_anchor = Q_st_bar

        # confidence score w^t_i
        wt_i, pseudo_label = torch.max(Q_anchor, 1)
        # confidence score w^s_j
        ws_j = torch.sum(Q_st_bar, 0)

        # filter by statistics mean
        uniformed_index = Q_st_bar.size(1)
        conf_label = torch.where(wt_i > 1/Q_st_bar.size(0), pseudo_label, uniformed_index)
        high_conf_label = conf_label.clone()
        source_private_label = torch.nonzero(ws_j < 1/Q_st_bar.size(1))
        for i in source_private_label:
            high_conf_label = torch.where(high_conf_label == i, uniformed_index, high_conf_label)
        high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)
        
        # for adaptive update
        new_beta = torch.sum(Q_st_bar,0).cpu().numpy()

        return high_conf_label_id, high_conf_label, conf_label, new_beta, pseudo_label, wt_i
        


class MemoryQueue(nn.Module):
    def __init__(self, feat_dim, batch_size, n_batch, T=0.05):
        super(MemoryQueue, self).__init__()
        self.feat_dim = feat_dim
        self.T = T
        self.batch_size = batch_size

        # init memory queue
        self.queue_size = batch_size * n_batch
        self.register_buffer('mem_feat', torch.zeros(self.queue_size, feat_dim))
        self.register_buffer('mem_id', torch.zeros((self.queue_size), dtype=int))
        # write pointer
        self.next_write = 0

    def forward(self, x):
        """
        obtain similarity between x and the features stored in memory queue
        """
        out = torch.mm(x, self.mem_feat.t()) / self.T
        return out

    def get_nearest_neighbor(self, anchors, id_anchors=None):
        """
        get anchors' nearest neighbor in memory queue 
        """
        # compute similarity first
        feat_mat = self.forward(anchors)

        # assign the similarity between features of the same sample with -1/T
        if id_anchors is not None:
            A = id_anchors.reshape(-1, 1).repeat(1, self.mem_id.size(0))
            B = self.mem_id.reshape(1, -1).repeat(id_anchors.size(0), 1)
            mask = torch.eq(A, B)
            id_mask = torch.nonzero(mask)
            temp = id_mask[:,1]
            feat_mat[:, temp] = -1 / self.T

        # obtain neighbor's similarity value and corresponding feature
        values, indices = torch.max(feat_mat, 1)
        nearest_feat = torch.zeros((anchors.size(0), self.feat_dim)).to(anchors.device)
        for i in range(anchors.size(0)):
            nearest_feat[i] = self.mem_feat[indices[i],:]
        return values, nearest_feat

    def update_queue(self, features, ids):
        """
        update memory queue
        """
        if features.size(0) != self.batch_size:
            return False
        w_ids = torch.arange(self.next_write, self.next_write+self.batch_size).to(features.device)
        self.mem_feat.index_copy_(0, w_ids, features.data)
        self.mem_id.index_copy_(0, w_ids, ids.data)
        self.mem_feat = F.normalize(self.mem_feat)

        # update write pointer
        self.next_write += self.batch_size
        if self.next_write == self.queue_size:
            self.next_write = 0

        return True

    def random_sample(self, size):
        """
        sample some features from memory queue randomly
        """ 
        id_t = torch.floor(torch.rand(size) * self.mem_feat.size(0)).long()
        sample_feat = self.mem_feat[id_t]
        return sample_feat
    

def sinkhorn(out, epsilon, sinkhorn_iterations):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()