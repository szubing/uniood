from abc import ABC, abstractmethod
import os
from tqdm import tqdm
import torch

from methods import method_classes
from methods import Oracle
from engine.scheduler import build_lr_scheduler
from engine.optimizer import build_optimizer
from engine.evaluator import UniDAEvaluator
from engine.dataloader import build_data_loaders
from tools.utils import get_save_scores_dir, get_save_dir, save_as_json


class DefaultTrainer(ABC):
    def __init__(self, cfg):
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = self.build_lr_scheduler(cfg, self.optimizer)

    def build_model(self, cfg):
        return method_classes[cfg.method](cfg)
    
    def build_optimizer(self, cfg, model):
        return build_optimizer(model, 
                               cfg.optimizer, 
                               cfg.base_lr, 
                               cfg.weight_decay, 
                               sgd_momentum=cfg.momentum, 
                               backbone_multiplier=cfg.backbone_multiplier,
                               clip_norm_value=cfg.clip_norm_value)
    
    def build_lr_scheduler(self, cfg, optimizer):
        return build_lr_scheduler(optimizer, 
                                  cfg.lr_scheduler, 
                                  cfg.warmup_iter, 
                                  cfg.max_iter, 
                                  warmup_type=cfg.warmup_type, 
                                  warmup_lr=cfg.warmup_min_lr)
    
    def _write_metrics(self, dict):
        print_str = ''
        for item in dict:
            print_str += f'{item}: {dict[item]}  '
        print(print_str)
    
    @abstractmethod
    def build_data_loaders(self, cfg):
        return
    
    @abstractmethod
    def build_evaluator(self, cfg):
        return
    
    @abstractmethod
    def train(self, cfg):
        return
    
    @abstractmethod
    def test(self, cfg):
        return
    
    @abstractmethod
    def load(self, checkpoint_path=None):
        return


class UniDaTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.source_data_loader, self.target_data_loader, \
                self.test_data_loader, self.val_data_loader = self.build_data_loaders(cfg)
        self.evaluator = self.build_evaluator(cfg)
        self.max_iter = cfg.max_iter
        self.cfg = cfg
        if self.use_features:
            self.model.backbone = torch.nn.Identity()
    
    def build_data_loaders(self, cfg):
        feature_dir = os.path.join(cfg.feature_dir, f'features-imgAug_{cfg.image_augmentation}', cfg.backbone.replace('/',''), cfg.dataset)
        source_feature_path = os.path.join(feature_dir, f'{cfg.source_domain}.pth')
        target_feature_path = os.path.join(feature_dir, f'{cfg.target_domain}.pth')
        if cfg.fixed_backbone and os.path.exists(source_feature_path) and os.path.exists(target_feature_path):
            self.use_features = True
            print('Use pretrained features as dataloader')
            cfg.num_workers = 0
        else:
            self.use_features = False
            print('Use I/O images as dataloader')
            source_feature_path = None
            target_feature_path = None

        return build_data_loaders(cfg.dataset, 
                                cfg.data_dir, 
                                cfg.source_domain, 
                                cfg.target_domain, 
                                cfg.n_share, 
                                cfg.n_source_private,
                                cfg.image_augmentation,
                                cfg.backbone,
                                cfg.no_balanced,
                                cfg.batch_size,
                                cfg.num_workers,
                                source_feature_path=source_feature_path,
                                target_feature_path=target_feature_path,
                                test_feature_path=target_feature_path,
                                val_feature_path=source_feature_path)
    
    def build_evaluator(self, cfg):
        n_source_classes = cfg.n_share + cfg.n_source_private
        return UniDAEvaluator(n_source_classes)
    
    def load(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.model.get_save_checkpoint_dir(self.cfg.fixed_backbone)
        
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=False)
    
    def train(self, cfg=None):
        self.model.before_training(cfg={'source_data_loader': self.source_data_loader,
                                        'target_data_loader': self.target_data_loader, 
                                        'test_data_loader': self.test_data_loader,
                                        'val_data_loader': self.val_data_loader})
        source_loader_iter = iter(self.source_data_loader)
        target_loader_iter = iter(self.target_data_loader)
        for step in range(self.max_iter):
            if self.model.require_source:
                try:
                    source_batch_datas = next(source_loader_iter)
                except StopIteration:
                    source_loader_iter = iter(self.source_data_loader)
                    source_batch_datas = next(source_loader_iter)
                
                source_images, source_labels = source_batch_datas['img'], source_batch_datas['label']
            else:
                source_images, source_labels= None, None
            
            if self.model.require_target:
                try:
                    target_batch_datas = next(target_loader_iter)
                except StopIteration:
                    target_loader_iter = iter(self.target_data_loader)
                    target_batch_datas = next(target_loader_iter)
                
                target_images, target_indexs = target_batch_datas['img'], target_batch_datas['idx']
            else:
                target_images, target_indexs = None, None

            batched_inputs = {'source_images': source_images, 
                              'source_labels': source_labels, 
                              'target_images': target_images,
                              'target_indexs': target_indexs}
            
            if isinstance(self.model, Oracle):
                batched_inputs['target_labels'] = target_batch_datas['label']
            
            self.model.before_forward()
            loss_dict = self.model(batched_inputs=batched_inputs)

            if loss_dict is not None:
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

                """
                If you need to accumulate gradients or do something similar, you can
                wrap the optimizer with your custom `zero_grad()` method.
                """
                self.optimizer.zero_grad()
                losses.backward()

                metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
                metrics_dict['step'] = f'{step}/{self.max_iter}'
                metrics_dict['lr'] = self.lr_scheduler.get_last_lr()[-1]
                self._write_metrics(metrics_dict)

                self.optimizer.step()
                self.lr_scheduler.step()
            
                self.model.after_backward()

                # if (step + 1) % 1000 == 0 and not self.cfg.fixed_backbone:
                #     self.test({'step': step + 1})
                #     self.model.before_training()

        self.model.after_training()

    def test(self, cfg=None):
        self.model.before_predict(cfg={'val_data_loader': self.val_data_loader, 
                                       'test_data_loader': self.test_data_loader})

        # evaluation on test data
        self.evaluator.reset() # remember to reset, this is very important
        if cfg is not None:
            current_step = cfg['step']
        else:
            current_step = 'final'
        logits = []
        iid_scores = []
        true_labels = []
        predict_labels = []
        predict_labels_without_ood = []
        for batch_datas in tqdm(self.test_data_loader):
            batched_inputs = {'test_images': batch_datas['img']}
            result_dict = self.model.predict(batched_inputs=batched_inputs)

            self.evaluator.process(batch_datas['label'], 
                                   result_dict['predict_labels'], 
                                   result_dict['predict_labels_without_ood'],
                                   result_dict['iid_scores'],
                                   result_dict['features'])
            
            true_labels.append(batch_datas['label'].cpu().detach())

            if result_dict['logits'] is not None:
                logits.append(result_dict['logits'].cpu().detach())
            if result_dict['iid_scores'] is not None:
                iid_scores.append(result_dict['iid_scores'].cpu().detach())
            if result_dict['predict_labels'] is not None:
                predict_labels.append(result_dict['predict_labels'].cpu().detach())
            if result_dict['predict_labels_without_ood'] is not None:
                predict_labels_without_ood.append(result_dict['predict_labels_without_ood'].cpu().detach())

        # save target logits/scores/prediction/ground_truth results
        if not self.cfg.eval_only:
            save_scores_pth = get_save_scores_dir(self.cfg.feature_dir, 
                                                f'{self.cfg.method}_{self.cfg.backbone}-{self.cfg.fixed_backbone}_{self.cfg.classifier_head}_{self.cfg.optimizer}_{self.cfg.batch_size}_{self.cfg.base_lr}_{self.cfg.fixed_BN}_{self.cfg.image_augmentation}_{current_step}-{self.cfg.max_iter}', 
                                                self.cfg.dataset, 
                                                self.cfg.source_domain, 
                                                self.cfg.target_domain, 
                                                self.cfg.n_share, 
                                                self.cfg.n_source_private, 
                                                self.cfg.seed,
                                                prefix='scores')
            
            save_data = {'true_labels': torch.cat(true_labels),
                        'target_logits': torch.cat(logits) if len(logits) != 0 else None,
                        'iid_scores': torch.cat(iid_scores) if len(iid_scores)!=0 else None, 
                        'predict_labels': torch.cat(predict_labels) if len(predict_labels)!=0 else None, 
                        'predict_labels_without_ood': torch.cat(predict_labels_without_ood) if len(predict_labels_without_ood)!=0 else None}
            torch.save(save_data, save_scores_pth)
        
        # save evaluation results
        results = self.evaluator.evaluate()
        self._write_metrics(results)

        save_dir = get_save_dir(self.cfg.result_dir, 
                                self.cfg.dataset, 
                                self.cfg.method,
                                self.cfg.source_domain, 
                                self.cfg.target_domain, 
                                self.cfg.n_share, 
                                self.cfg.n_source_private,
                                self.cfg.backbone,
                                self.cfg.optimizer,
                                self.cfg.base_lr, 
                                self.cfg.classifier_head,
                                self.cfg.fixed_backbone,
                                self.cfg.fixed_BN,
                                self.cfg.image_augmentation,
                                self.cfg.batch_size,
                                f'{current_step}-{self.cfg.max_iter}',
                                self.cfg.seed)
        
        save_as_json(results, save_dir)
        


            
