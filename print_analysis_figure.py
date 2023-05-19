import faiss
import numpy as np
import os
import copy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from configs.default import *
from engine.evaluator import UniDAEvaluator
from datasets import dataset_classes
from tools.utils import get_save_logits_dir, get_save_dir, get_save_scores_dir, load_json

DATASETS = ['office31', 'officehome', 'visda', 'domainnet']
NAME_DATA = {'office31': 'Office', 'officehome': 'OfficeHome', 'visda': 'VisDA', 'domainnet': 'DomainNet'}
METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT']

BACKBONES = ['dinov2_vitl14', 'ViT-L/14@336px']
SETTINGS = ['open-partial', 'open', 'closed', 'partial']
# SETTINGS = ['open-partial']
THRESHOD_NUM = 20
SEEDS = [1, 2, 3]

METRIC = 'H3-score' # 'H3-score'
if METRIC == 'H3-score':
    METRIC_name = 'H$^3$-score'
else:
    METRIC_name = METRIC


DOMAINS = {'office31': ['amazon', 'dslr', 'webcam'], 
           'officehome': ['Art', 'Clipart', 'Product', 'RealWorld'],
           'visda': ['syn', 'real'],
           'domainnet': ['painting', 'real', 'sketch']}

NN = {'open-partial': {'office31': [10, 10],
                        'officehome': [10, 5],
                        'visda': [6, 3],
                        'domainnet': [150, 50]},

        'open': {'office31': [10, 0],
                        'officehome': [15, 0],
                        'visda': [6, 0],
                        'domainnet': [150, 0]},

        'closed': {'office31': [31, 0],
                        'officehome': [65, 0],
                        'visda': [12, 0],
                        'domainnet': [345, 0]},

        'partial': {'office31': [10, 21],
                        'officehome': [25, 40],
                        'visda': [6, 6],
                        'domainnet': [150, 195]}
}

MAX_ITERS = {'open-partial': {'office31': 5000,
                            'officehome': 5000,
                            'visda': 10000,
                            'domainnet': 10000},

                'open': {'office31': 5000,
                            'officehome': 5000,
                            'visda': 10000,
                            'domainnet': 10000},

                'closed': {'office31': 10000,
                            'officehome': 10000,
                            'visda': 20000,
                            'domainnet': 20000}, 
                
                'partial': {'office31': 10000,
                            'officehome': 10000,
                            'visda': 20000,
                            'domainnet': 20000},                 
}


def get_entropy_from_logits(logits):
    probs = F.softmax(logits, dim=1)
    entropy_values = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
    return entropy_values

for backbone in BACKBONES:
    for method in METHODS:
        if len(SETTINGS) == 1:
            fig = plt.figure()
        for dataset in DATASETS:
            results = []
            if len(SETTINGS) != 1:
                fig = plt.figure()

            for setting in SETTINGS:
                if setting in ('closed', 'partial') and dataset == 'domainnet':
                    continue
                if setting in ('closed', 'partial'):
                    metric = 'AA'
                else:
                    metric = METRIC

                result_task = np.zeros((len(SEEDS), THRESHOD_NUM+1))
                task_num = 0
                for source_domain in DOMAINS[dataset]:
                    for target_domain in DOMAINS[dataset]:
                        if source_domain != target_domain and not (source_domain == 'real' and target_domain == 'syn'):
                            n_share, n_source_private = NN[setting][dataset][0], NN[setting][dataset][1]
                            data = dataset_classes[dataset](DATA_DIR, source_domain, target_domain, n_share, n_source_private)
                            labels = torch.tensor([item['label'] for item in data.test])
                            evaluator = UniDAEvaluator(n_share + n_source_private)

                            result_seed = np.zeros((len(SEEDS), THRESHOD_NUM+1))
                            seed_id = 0
                            for seed in SEEDS:
                                # get saved logits
                                save_logits_pth = get_save_logits_dir(FEATURE_DIR, 
                                                        f'{method}_{backbone}-True_prototype_sgd_32_0.01_False_none_final-{MAX_ITERS[setting][dataset]}', 
                                                        dataset, 
                                                        source_domain, 
                                                        target_domain, 
                                                        n_share,
                                                        n_source_private,
                                                        seed)
        
                                logits = torch.load(save_logits_pth)
                                max_logits, predict_labels = torch.max(logits, -1)
                                probs = F.softmax(logits, dim=1)
                                max_probs, _ = torch.max(probs, -1)
                                entropy_values = get_entropy_from_logits(logits)

                                # get save scores
                                save_scores_pth = get_save_scores_dir(FEATURE_DIR, 
                                                        f'{method}_{backbone}-True_prototype_sgd_32_0.01_False_none_final-{MAX_ITERS[setting][dataset]}',  
                                                        dataset, 
                                                        source_domain, 
                                                        target_domain, 
                                                        n_share, 
                                                        n_source_private, 
                                                        seed)
                                    
                                iid_scores = torch.load(save_scores_pth)

                                if isinstance(iid_scores, dict):
                                    predict_labels = iid_scores['predict_labels_without_ood']
                                    iid_scores = iid_scores['iid_scores']

                                # normalize iid_scores to 0-1
                                if method in ('DANCE', 'SO'):
                                    iid_scores = iid_scores / torch.log(torch.tensor(n_share+n_source_private)) + 1.0
                                elif method == 'UniOT':
                                    iid_scores = iid_scores * len(iid_scores) / 2.0

                                # get saved original results
                                save_dir = get_save_dir(RESULT_DIR, 
                                                        dataset, 
                                                        method,
                                                        source_domain, 
                                                        target_domain, 
                                                        n_share, 
                                                        n_source_private,
                                                        backbone,
                                                        'sgd',
                                                        '0.01', 
                                                        'prototype',
                                                        True,
                                                        False,
                                                        'none',
                                                        32,
                                                        f'final-{MAX_ITERS[setting][dataset]}',
                                                        seed)
                                result_original = load_json(save_dir)


                                # re-evaluate the results by different choice of iid_scores or ood detection method.
                                for iid_threshold in range(THRESHOD_NUM + 1):
                                    # threshold = iid_threshold / THRESHOD_NUM * (iid_scores.max() - iid_scores.min()) + iid_scores.min()
                                    threshold = iid_threshold / THRESHOD_NUM

                                    ood_indexs = iid_scores < threshold
                                    grd_labels  = copy.deepcopy(labels)
                                    prd_labels = copy.deepcopy(predict_labels)
                                    prd_labels[ood_indexs] = n_share + n_source_private
                                    evaluator.reset()
                                    evaluator.process(grd_labels, prd_labels)
                                    result_current = evaluator.evaluate()  
                                    if metric == 'H3-score':
                                        nmi = result_original['NMI']
                                        recall_avg_auc = np.array(result_current['Recall-all'])
                                        acc_shared = recall_avg_auc[:-1].mean()
                                        acc_ood = recall_avg_auc[-1]
                                        acc = 3*(1/(1/(acc_shared+1e-5) + 1/(acc_ood+1e-5) + 1/(nmi+1e-5)))
                                    else:
                                        acc = result_current[metric]

                                    result_seed[seed_id, iid_threshold] = float(acc)
                                
                                seed_id += 1

                            task_num += 1
                            result_task += result_seed

                result_task /= task_num
                plt.errorbar(np.arange(THRESHOD_NUM + 1)/THRESHOD_NUM, result_task.mean(axis=0), yerr=result_task.std(axis=0), label=NAME_DATA[dataset] if len(SETTINGS) == 1 else f'({n_share}/{n_source_private})')
            
            
            if len(SETTINGS) != 1:
                save_file_path = os.path.join('figures', dataset, backbone.replace('/',''), f'{method}-{METRIC}.pdf')

                result_dir = os.path.dirname(save_file_path)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                
                plt.xlabel('Normalized threshold', fontdict={'size': 16})
                plt.ylabel(METRIC_name, fontdict={'size': 16})
                plt.yticks(np.arange(0,101,20))
                # plt.title('error bar')
                plt.legend(prop={'size': 12})
                plt.tick_params(labelsize=12)
                plt.savefig(save_file_path)
                plt.close()
        
        if len(SETTINGS) == 1:
            save_file_path = os.path.join('figures', backbone.replace('/',''), f'{method}-{SETTINGS[0]}-{METRIC}.pdf')
            result_dir = os.path.dirname(save_file_path)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            plt.xlabel('Normalized threshold', fontdict={'size': 16})
            plt.ylabel(METRIC_name, fontdict={'size': 16})
            plt.yticks(np.arange(0,101,20))
            # plt.title('error bar')
            plt.legend(prop={'size': 12})
            plt.tick_params(labelsize=12)
            plt.savefig(save_file_path)
            plt.close()

