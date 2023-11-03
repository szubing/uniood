import faiss # must be first imported in the main.py --> I don't know why
import os
import csv
import numpy as np

from tools.utils import get_save_dir, load_json
from configs import default
from models import backbone_names, head_names

import argparse

parser = argparse.ArgumentParser()

DATASETS = ['office31', 'officehome', 'visda', 'domainnet']

METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT', 'WiSE-FT', 'ClipCrossModel', 'ClipZeroShot', 'ClipDistill', 'ClipDistillTemp1.0', 'AutoDistill']
# METHODS = ['SO', 'SO3.0', 'SO1.5', 'ClipDistill', 'ClipDistill3.0', 'ClipDistill1.5']
# METHODS += ['SOMarginP', 'ClipDistillMarginP']
# METHODS += ['FocalDistill']
METHODS += ['Auto_only_cal', 'Auto_wo_iid', 'Auto_wo_nll', 'Auto_wo_ood']
# METHODS = ['AutoDistill']

# METHODS += ['debug0.1', 'debug0.2', 'debug0.3', 'debug0.4', 'debug0.5', 'debug0.6', 'debug0.7', 'debug0.8', 'debug0.9', 'debug1.0'] # for hyperparameters analysis for CLIP distillation

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

# setting
STETTING = 'closed'
# ITERS = [1000 * (i+1) for i in range(10)]
ITERS = ['final']

SEEDS = [1, 2, 3]

RESULTS_TERMS = ['AA', 'H-score', 'H3-score', 'AUROC', 'OSCR']

if STETTING == 'closed' or STETTING == 'partial':
    RESULTS_TERMS = ['OA', 'AA', 'Closed-set OA', 'Closed-set AA']


def main(args):
    global METHODS, MAX_ITERS, STETTING

    # args.backbone = 'dinov2_vitl14'
    # args.backbone = 'resnet50'
    # args.backbone = 'ViT-B/16'
    # args.suffix = '_my'
    save_result_dir = 'results'
    if args.suffix is not None:
        args.result_dir = args.result_dir + args.suffix
        save_result_dir += args.suffix

    result_dir_resnet50 = None
    args.fixed_backbone = True
    if not 'ViT' in args.backbone:
        METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT', 'ClipDistill', 'AutoDistill']
    if args.backbone == 'resnet50':
        args.fixed_backbone = False
        assert STETTING == 'open-partial'
        result_dir_resnet50 = '/data1/deng.bin/coding/uniood/experiments-0425'
        
    overall_mean_steps = []
    for step in ITERS:
        overall_mean, all_columns_mean, all_columns_std = [], [], []
        for dataset in DATASETS:
            for method in METHODS:
                # if method in ['SOMarginP', 'ClipDistillMarginP']:
                #     args.result_dir = './experimentsMarginP'
                # else:
                #     args.result_dir = './experiments'
                max_iter = MAX_ITERS[STETTING][dataset]
                if args.backbone == 'resnet50' and method in ('SO', 'DANCE', 'OVANet', 'UniOT'):
                    max_iter = 10000

                classifier_head = args.classifier_head
                result_data_method = {}
                for source_domain in DOMAINS[dataset]:
                    for target_domain in DOMAINS[dataset]:
                        if source_domain != target_domain and not (source_domain == 'real' and target_domain == 'syn'):
                            all_seed_dict = {}
                            for seed in SEEDS:
                                if method == 'ClipZeroShot':
                                    seed_ = 1
                                else:
                                    seed_ = seed
                                
                                save_dir = get_save_dir(result_dir_resnet50 if (result_dir_resnet50 is not None and method in ('SO', 'DANCE', 'OVANet', 'UniOT')) else args.result_dir, 
                                                        dataset, 
                                                        method,
                                                        source_domain, 
                                                        target_domain, 
                                                        NN[STETTING][dataset][0], 
                                                        NN[STETTING][dataset][1],
                                                        args.backbone,
                                                        args.optimizer,
                                                        args.base_lr, 
                                                        classifier_head,
                                                        args.fixed_backbone,
                                                        args.fixed_BN,
                                                        args.image_augmentation,
                                                        args.batch_size,
                                                        f'{step}-{max_iter}',
                                                        seed_)
                                result = load_json(save_dir)
                                if result is not None:
                                    all_seed_dict[seed] = result
                                else:
                                    print(save_dir, ' not exists')

                            result_mean, result_std = average_seed(all_seed_dict)
                            result_data_method[f'{source_domain}-{target_domain}'] = result_mean
                            all_columns_mean.append([dataset, NN[STETTING][dataset][0], NN[STETTING][dataset][1], method, source_domain, target_domain] + [result_mean[term] for term in RESULTS_TERMS])
                            all_columns_std.append([dataset, NN[STETTING][dataset][0], NN[STETTING][dataset][1], method, source_domain, target_domain] + [result_std[term] for term in RESULTS_TERMS])

                mean_data_method, _ = average_seed(result_data_method)
                overall_mean.append([dataset, NN[STETTING][dataset][0], NN[STETTING][dataset][1], method] + [mean_data_method[term] for term in RESULTS_TERMS])
                overall_mean_steps.append([dataset, NN[STETTING][dataset][0], NN[STETTING][dataset][1], method, step] + [mean_data_method[term] for term in RESULTS_TERMS]) 

        all_headers = ['dataset', 'n_share', 'n_source_private', 'method', 'source', 'target'] + RESULTS_TERMS
        result_path = f'{args.backbone}-{args.optimizer}-{args.base_lr}-{args.classifier_head}-{args.fixed_backbone}-{args.fixed_BN}-{args.image_augmentation}-{args.batch_size}'.replace('/','')
        save_all_csv(all_headers, all_columns_mean, os.path.join(save_result_dir, STETTING, f'{step}', result_path, 'mean.csv'))
        save_all_csv(all_headers, all_columns_std, os.path.join(save_result_dir, STETTING, f'{step}', result_path, 'std.csv'))

        overall_headers = ['dataset', 'n_share', 'n_source_private', 'method'] + RESULTS_TERMS
        save_all_csv(overall_headers, overall_mean, os.path.join(save_result_dir, STETTING, f'{step}', result_path, 'mean_average.csv'))

    overall_headers_steps = ['dataset', 'n_share', 'n_source_private', 'method', 'step'] + RESULTS_TERMS
    save_all_csv(overall_headers_steps, overall_mean_steps, os.path.join(save_result_dir, STETTING, result_path, 'mean_average.csv'))


def average_seed(all_seed_dict):
    result_mean = {}
    result_std = {}
    for term in RESULTS_TERMS:
        result = []
        for seed in all_seed_dict.keys():
            if term in all_seed_dict[seed].keys():
                result.append(all_seed_dict[seed][term])
            elif term == 'Closed-set OA':
                result.append(all_seed_dict[seed]['Closed Set Accuracy']['OA'])
            elif term == 'Closed-set AA':
                result.append(all_seed_dict[seed]['Closed Set Accuracy']['AA'])
            else:
                result.append(all_seed_dict[seed]['OSR Accuracy'][term])
        result_mean[term] = round(np.mean(result), 2)
        result_std[term] = round(np.std(result), 2)
    
    return result_mean, result_std

def save_all_csv(all_headers, all_columns, result_path):
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(result_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(all_columns)


if __name__ == "__main__":
    ###########################
    # Directory Config (modify if using your own paths)
    ###########################
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default.DATA_DIR,
        help="where the dataset is saved",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=default.FEATURE_DIR,
        help="where to save pre-extracted features",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=default.RESULT_DIR,
        help="where to save experiment results",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="the save_file suffix after result_dir",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-L/14@336px",
        choices=backbone_names,
        help="specify the encoder-backbone to use",
    )
    parser.add_argument(
        "--classifier_head",
        type=str,
        default="prototype",
        choices=head_names,
        help="classifier head architecture",
    )
    parser.add_argument(
        "--fixed_backbone",
        action="store_true",
        help="wheather fixed backbone during training",
    )
    parser.add_argument(
        "--fixed_BN",
        action="store_true",
        help="wheather fixed batch normalization layers during training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size for test (feature extraction and evaluation)",
    )
    parser.add_argument(
        "--image_augmentation",
        type=str,
        default='none',
        choices=['none', # only a single center crop
                'flip', # add random flip view
                'randomcrop', # add random crop view
                ],
        help="specify the image augmentation to use.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adamw", "sgd"],
        help="optimizer"
    )
    parser.add_argument(
        '--base_lr', 
        type=float, 
        default=1e-2,
        help='base learning rate'
    )
    args = parser.parse_args()
    main(args)