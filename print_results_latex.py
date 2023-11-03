import os
import argparse
import csv
import pandas as pd
import numpy as np

from models import head_names

parser = argparse.ArgumentParser()

DATASETS = ['office31', 'officehome', 'visda', 'domainnet']

# METHODS = ['SO', 'DANCE', 'OVANet', 'UniOT', 'WiSE-FT', 'ClipCrossModel', 'ClipZeroShot']
# METHODS += ['ClipDistillTemp1.0', 'AutoDistill']
# METHODS += ['Auto_only_cal']
# METHODS = ['Auto_wo_iid', 'Auto_wo_nll', 'Auto_wo_ood', 'AutoDistill']
METHODS = ['AutoDistill']

# SETTINGS = ['open-partial', 'open', 'closed', 'partial']
# SETTINGS = ['open-partial']
SETTINGS = ['partial']

METRIC = 'H-score' # 'OSCR'
# BACKBONES = ['resnet50', 'dinov2_vitl14', 'ViT-L/14@336px']
BACKBONES = ['ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
# BACKBONES = ['dinov2_vitl14']
STEP = 'final'

DIR = 'results'

header_csv_latex = ['method', '&']
for backbone in BACKBONES:
    for dataset in DATASETS:
        for setting in SETTINGS:
            # if dataset == 'domainnet' and setting in ('closed', 'partial'):
            #     continue
            header_csv_latex.append(backbone + dataset + setting)
            header_csv_latex.append('&')
        
    header_csv_latex.append('Avg')
    header_csv_latex.append('&')

header_csv_latex[-1] = '\\\\'

def main(args):
    data_csv_latex = []
    for method in METHODS:
        method_csv = []
        for backbone in BACKBONES:           
            backbone_csv = []
            if backbone == 'resnet50':
                fixed_backbone = False
            else:
                fixed_backbone = True

            for dataset in DATASETS:
                for setting in SETTINGS:
                    # if dataset == 'domainnet' and setting in ('closed', 'partial'):
                    #     continue
                    if METRIC in ('H-score', 'H3-score') and setting in ('closed', 'partial'):
                        metric = 'AA'
                    elif METRIC == 'OSCR' and setting in ('closed', 'partial'):
                        metric = 'Closed-set OA'
                    else:
                        metric = METRIC

                    if metric == 'OSCR' and method == 'ClipZeroShot':
                        DIR = 'results_old'
                    else:
                        DIR = 'results'

                    if backbone in ('resnet50', 'dinov2_vitl14') and method in ('WiSE-FT', 'ClipCrossModel', 'ClipZeroShot', 'ClipDistillTemp1.0', 'AutoDistill'):
                        method_csv.append('-')
                        backbone_csv.append('-')
                        method_csv.append('&')
                    else:
                        result_path = f'{backbone}-{args.optimizer}-{args.base_lr}-{args.classifier_head}-{fixed_backbone}-{args.fixed_BN}-{args.image_augmentation}-{args.batch_size}'.replace('/','')
                        path_load = os.path.join(DIR, setting, f'{STEP}', result_path, 'mean_average.csv')
                        df = pd.read_csv(path_load)
                        result = df[(df['method'] == method) & (df['dataset'] == dataset)][metric]
                        method_csv.append(float(result))
                        backbone_csv.append(float(result))
                        method_csv.append('&')

            if backbone in ('resnet50', 'dinov2_vitl14') and method in ('WiSE-FT', 'ClipCrossModel', 'ClipZeroShot', 'ClipDistillTemp1.0', 'AutoDistill'):
                method_csv.append('-')
            else:
                method_csv.append(round(float(np.mean(backbone_csv)),2))

            method_csv.append('&')

        method_csv[-1] = '\\\\'
    
        data_csv_latex.append([method, '&'] + method_csv)

    if len(BACKBONES) == 1:
        bname = BACKBONES[0].replace('/','')
    else:
        assert len(SETTINGS) == 1
        bname = 'mix'

    save_path_latex = os.path.join('latex', 'result', f'{METRIC}-{bname}-{STEP}_latex.csv')
    data_csv_latex = process_list_with_boldtext(data_csv_latex)
    save_all_csv(header_csv_latex, data_csv_latex, save_path_latex)



def save_all_csv(all_headers, all_columns, result_path):
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(result_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(all_columns)


def process_list_with_boldtext(data, row_bold=False, latex=True):
    row_num = len(data)
    column_num = len(data[0])
    import copy
    data_copy = copy.deepcopy(data)
    p = 2 if latex else 1
    if row_bold:
        for i in range(row_num):
            value = -1
            for j in range(column_num):
                if j % p == 0 and isinstance(data[i][j], float):
                    if value <= data[i][j]:
                        maxJ = j
                        value = data[i][j]

            data_copy[i][maxJ] = '\\textbf{' + f'{data[i][maxJ]}' + '}'
    else:
        for j in range(column_num):
            value = -1
            maxI = None
            for i in range(row_num):
                if j % p == 0 and isinstance(data[i][j], float):
                    if value <= data[i][j]:
                        maxI = i
                        value = data[i][j]
                else:
                    maxI = None if maxI is None else maxI

            if maxI is not None:
                data_copy[maxI][j] = '\\textbf{' + f'{data[maxI][j]}' + '}'
    
    return data_copy






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
