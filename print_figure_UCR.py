import faiss
import numpy as np
import os
import copy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from configs.default import *
from datasets import dataset_classes
from tools.utils import get_save_logits_dir, get_save_dir, get_save_scores_dir, load_json

dataset = 'visda'
source_domain = 'syn'
target_domain = 'real'
# METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT', 'WiSE-FT', 'ClipCrossModel', 'ClipZeroShot', 'debug0.3']
method_name = {'SO': 'SO', 'DANCE': 'DANCE', 'OVANet': 'OVANet', 'UniOT': 'UniOT', 'WiSE-FT': 'WiSE-FT', 'ClipCrossModel': 'CLIP cross-model', 'ClipZeroShot': 'CLIP zero-shot', 'debug0.3': 'CLIP distillation'}

BACKBONES = ['dinov2_vitl14', 'ViT-L/14@336px']
n_share = 6
n_source_private = 3
seed = 1

METRIC_name = 'UCR'


def compute_oscr(x1, x2, pred_in, labels_in):
    m_x1 = np.zeros(len(x1))
    m_x1[pred_in == labels_in] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values
    
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)] 

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n-1):
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

        OSCR = OSCR + h*w

    return OSCR, FPR, CCR


for backbone in BACKBONES:
    if backbone == 'dinov2_vitl14':
        METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT',]
    else:
        METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT', 'WiSE-FT', 'ClipCrossModel', 'ClipZeroShot', 'debug0.3']
    backbone_name = backbone.replace('/','')
    fig = plt.figure()
    for method in METHODS:
        data = dataset_classes[dataset](DATA_DIR, source_domain, target_domain, n_share, n_source_private)
        labels = torch.tensor([item['label'] for item in data.test])

        seed_id = 0
        save_logits_pth = get_save_logits_dir(FEATURE_DIR, 
                                f'{method}_{backbone}-True_prototype_sgd_32_0.01_False_none_final-10000', 
                                dataset, 
                                source_domain, 
                                target_domain, 
                                n_share,
                                n_source_private,
                                seed)

        logits = torch.load(save_logits_pth)
        max_logits, predict_labels = torch.max(logits, -1)

        # get save scores
        save_scores_pth = get_save_scores_dir(FEATURE_DIR, 
                                f'{method}_{backbone}-True_prototype_sgd_32_0.01_False_none_final-10000',  
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

        label_set = set(labels.tolist())
        private_label_set = label_set - set(range(n_share+n_source_private))
        target_private_indexs = [True if lab in private_label_set else False for lab in labels.tolist()]
        shared_indexs = [False if id else True for id in target_private_indexs]
        x1, x2 = iid_scores[shared_indexs], iid_scores[target_private_indexs]
        # OSCR
        oscr_socre, fpr, ccr = compute_oscr(x1, x2, predict_labels[shared_indexs], labels[shared_indexs])
        
        plt.plot(fpr[:-1], ccr[:-1], label=f'{method_name[method]}')
        
        
    save_file_path = os.path.join('figures', 'UCR', f'{dataset}-{backbone_name}-{method}.pdf')

    result_dir = os.path.dirname(save_file_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    plt.xlabel('False Positive Rate', fontdict={'size': 16})
    plt.ylabel('Correct Classification Rate', fontdict={'size': 16})
    plt.legend(prop={'size': 12})
    plt.tick_params(labelsize=12)
    plt.xlim(0,1)
    plt.ylim(0,)
    plt.savefig(save_file_path)
    plt.close()



