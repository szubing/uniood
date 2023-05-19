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
from tools.utils import get_save_dir, load_json, get_save_logits_dir


DATASETS = ['office31', 'officehome', 'visda', 'domainnet']
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
SEEDS = [1,2,3]
backbone = 'ViT-L/14@336px'
setting = 'open-partial'
temps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for dataset in DATASETS:
    result_task_ours = np.zeros((len(SEEDS), len(temps)))
    result_task_zeroshot = np.zeros((len(SEEDS), len(temps)))
    result_task_ours_MLS = np.zeros((len(SEEDS), len(temps)))
    result_task_ours_MSP = np.zeros((len(SEEDS), len(temps)))
    temp_id = 0
    for temp in temps:
        method = f'debug{temp}' if temp != 0.5 else 'debug'
        task_id = 0
        for source_domain in DOMAINS[dataset]:
            for target_domain in DOMAINS[dataset]:
                if source_domain != target_domain and not (source_domain == 'real' and target_domain == 'syn'):

                    n_share, n_source_private = NN[setting][dataset][0], NN[setting][dataset][1]
                    data = dataset_classes[dataset](DATA_DIR, source_domain, target_domain, n_share, n_source_private)
                    labels = torch.tensor([item['label'] for item in data.test])
                    evaluator = UniDAEvaluator(n_share + n_source_private)

                    seed_id = 0
                    for seed in SEEDS:
                        save_dir = get_save_dir('./experiments', 
                                                dataset, 
                                                method,
                                                source_domain, 
                                                target_domain, 
                                                NN[setting][dataset][0], 
                                                NN[setting][dataset][1],
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
                        result = load_json(save_dir)
                        result_task_ours[seed_id, temp_id] += float(result['OSR Accuracy']['OSCR'])

                        save_dir = get_save_dir('./experiments', 
                                                dataset, 
                                                'ClipZeroShot',
                                                source_domain, 
                                                target_domain, 
                                                NN[setting][dataset][0], 
                                                NN[setting][dataset][1],
                                                backbone,
                                                'sgd',
                                                '0.01', 
                                                'prototype',
                                                True,
                                                False,
                                                'none',
                                                32,
                                                f'final-{MAX_ITERS[setting][dataset]}',
                                                1)
                        result = load_json(save_dir)
                        result_task_zeroshot[seed_id, temp_id] += float(result['OSR Accuracy']['OSCR'])


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
                        # entropy_values = get_entropy_from_logits(logits)
                        grd_labels  = copy.deepcopy(labels)
                        prd_labels_without_ood = copy.deepcopy(predict_labels)
                        prd_labels = copy.deepcopy(predict_labels)
                        ood_indexs = max_probs < 0.5
                        prd_labels[ood_indexs] = n_share + n_source_private
                        evaluator.reset()
                        evaluator.process(labels, prd_labels, prd_labels_without_ood, max_logits)
                        result_current = evaluator.evaluate()
                        result_task_ours_MLS[seed_id, temp_id] += result_current['OSR Accuracy']['OSCR']

                        evaluator.reset()
                        evaluator.process(labels, prd_labels, prd_labels_without_ood, max_probs)
                        result_current = evaluator.evaluate()
                        result_task_ours_MSP[seed_id, temp_id] += result_current['OSR Accuracy']['OSCR']


                        seed_id += 1
                    
                    task_id += 1
        temp_id += 1

    result_task_zeroshot /= task_id
    result_task_ours /= task_id
    result_task_ours_MSP /= task_id
    result_task_ours_MLS /= task_id

    fig = plt.figure()
    plt.errorbar(temps, result_task_zeroshot.mean(axis=0), yerr=result_task_zeroshot.std(axis=0), label='CLIP zero-shot')
    plt.errorbar(temps, result_task_ours.mean(axis=0), yerr=result_task_ours.std(axis=0), label='CLIP distillation (Ours)')
    plt.errorbar(temps, result_task_ours_MSP.mean(axis=0), yerr=result_task_ours_MSP.std(axis=0), label='CLIP distillation (Ours, MSP)')
    plt.errorbar(temps, result_task_ours_MLS.mean(axis=0), yerr=result_task_ours_MLS.std(axis=0), label='CLIP distillation (Ours, MLS)')


    save_file_path = os.path.join('figures', 'hypars', f'{dataset}-ViTL14@336px-ours.pdf')

    result_dir = os.path.dirname(save_file_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    plt.xlabel('Temperature', fontdict={'size': 16})
    plt.ylabel('UCR', fontdict={'size': 16})
    if dataset in ('office31','officehome'):
        plt.yticks(np.arange(85,96,5))
    elif dataset == 'visda':
        plt.yticks(np.arange(75,86,5))
    elif dataset == 'domainnet':
        plt.yticks(np.arange(60,81,5))
    plt.xticks(temps)
    plt.legend(prop={'size': 12})
    plt.tick_params(labelsize=12)
    plt.savefig(save_file_path)
    plt.close()




