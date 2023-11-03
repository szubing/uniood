import os
import argparse
import csv
import pandas as pd
import numpy as np

from models import head_names

parser = argparse.ArgumentParser()

DATASETS = ['office31', 'officehome', 'visda', 'domainnet']

DOMAINS = {'office31': ['amazon', 'dslr', 'webcam'], 
           'officehome': ['Art', 'Clipart', 'Product', 'RealWorld'],
           'visda': ['syn', 'real'],
           'domainnet': ['painting', 'real', 'sketch']}
domain_name_map = {'amazon':'A', 'dslr':'D', 'webcam':'W',
                   'Art':'A', 'Clipart':'C', 'Product':'P', 'RealWorld':'R',
                   'syn':'S',
                   'painting':'P', 'real':'R', 'sketch':'S'}

# METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT', 'WiSE-FT', 'ClipCrossModel', 'ClipZeroShot', 'AutoDistill']
METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT', 'ClipDistill']
method_name_map = {'SO':'SO', 'DANCE': 'DANCE', 'OVANet':'OVANet', 'UniOT':'UniOT', 'WiSE-FT':'WiSE-FT', 'ClipCrossModel':'CLIP cross-model', 'ClipZeroShot':'CLIP zero-shot', 'AutoDistill':'CLIP distillation (Ours)', 'ClipDistill':'CLIP distillation (Ours)'}

# SETTINGS = ['open-partial', 'open', 'closed', 'partial']
SETTINGS = ['open-partial']

METRICS = ['H-score', 'H3-score', 'OSCR']

# BACKBONES = ['resnet50', 'dinov2_vitl14', 'ViT-L/14@336px']
# BACKBONES = ['ViT-L/14@336px']
# BACKBONES = ['dinov2_vitl14']
BACKBONES = ['resnet50']

STEP = 'final'

DIR = 'results'

def main(args):
    for setting in SETTINGS:
        for dataset in DATASETS:
            # if dataset == 'domainnet' and setting in ('closed','partial'):
            #     continue
            for backbone in BACKBONES:
                backbone_name = backbone.replace('/', '')
                if backbone == 'resnet50':
                    fixed_backbone = False
                    if setting != 'open-partial':
                        continue
                else:
                    fixed_backbone = True
                for metric in METRICS:
                    if metric in ('H-score', 'H3-score') and setting in ('closed', 'partial'):
                        metric_ = 'AA'
                    elif metric == 'OSCR' and setting in ('closed', 'partial'):
                        metric_ = 'Closed-set OA'
                    else:
                        metric_ = metric

                    method_csv = []
                    method_names = []
                    for method in METHODS:
                        if metric == 'OSCR' and method == 'ClipZeroShot':
                            DIR = 'results_old'
                        else:
                            DIR = 'results'
                                    
                        if backbone in ('resnet50', 'dinov2_vitl14') and method in ('WiSE-FT', 'ClipCrossModel', 'ClipZeroShot', 'AutoDistill'):
                            continue
                        # if metric in ('H-score', 'H3-score') and method in ('WiSE-FT', 'ClipCrossModel', 'ClipZeroShot', 'debug0.3'):
                        #     continue

                        result_path = f'{backbone_name}-{args.optimizer}-{args.base_lr}-{args.classifier_head}-{fixed_backbone}-{args.fixed_BN}-{args.image_augmentation}-{args.batch_size}'
                        path_load_mean = os.path.join(DIR, setting, f'{STEP}', result_path, 'mean.csv')
                        df_mean = pd.read_csv(path_load_mean)
                        path_load_std = os.path.join(DIR, setting, f'{STEP}', result_path, 'std.csv')
                        df_std = pd.read_csv(path_load_std)
                        domain_csv = []
                        domain_csv_head = []
                        for source_domain in DOMAINS[dataset]:
                            for target_domain in DOMAINS[dataset]:
                                if source_domain != target_domain and not (source_domain == 'real' and target_domain == 'syn'):
                                    result_mean = df_mean[(df_mean['method'] == method) & (df_mean['dataset'] == dataset) & (df_mean['source'] == source_domain) & (df_mean['target'] ==        target_domain)][metric_]

                                    result_std = df_std[(df_std['method'] == method) & (df_std['dataset'] == dataset) & (df_std['source'] == source_domain) & (df_std['target'] ==        target_domain)][metric_]

                                    domain_csv_head += [f'{domain_name_map[source_domain]}2{domain_name_map[target_domain]}']
                                    domain_csv += [float(result_mean), float(result_std)]

                        method_names += [method_name_map[method]]
                        domain_csv += [round(float(np.mean(domain_csv[0::2])),2)]
                        method_csv.append(domain_csv)
                        # domain_csv_head += ['Avg']
                        # method_csv.append(float(np.mean(domain_csv[0::2])))

                    max_id = np.array(method_csv).argmax(axis=0)

                    save_csv = [[name, '&'] for name in method_names]
                    save_csv_head = ['Methods', '&']
                    for i in range(len(domain_csv_head)):
                        save_csv_head += [domain_csv_head[i], '&']
                        for j in range(len(method_csv)):
                            if j == max_id[2*i]:
                                save_csv[j] += ['\\textbf{'+f'{method_csv[j][2*i]}'+ '}$\\pm$' + f'{method_csv[j][2*i+1]}', '&']
                            else:
                                save_csv[j] += [f'{method_csv[j][2*i]}$\\pm${method_csv[j][2*i+1]}', '&']
                    
                    i = len(domain_csv_head)
                    save_csv_head += ['Avg', '&']
                    for j in range(len(method_csv)):
                        if j == max_id[2*i]:
                            save_csv[j] += ['\\textbf{'+f'{method_csv[j][2*i]}'+ '}', '&']
                        else:
                            save_csv[j] += [f'{method_csv[j][2*i]}', '&']
                    
                    save_csv_head[-1] = '\\\\'
                    for j in range(len(method_csv)):
                        save_csv[j][-1] = '\\\\'
                    
                    save_path = os.path.join('latex', 'appendix', setting, dataset, backbone, f'{metric}.csv')

                    save_all_csv(save_csv_head, save_csv, save_path)


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
