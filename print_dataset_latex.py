import csv
import os
from collections import Counter
import numpy as np

from datasets import dataset_classes

from configs.default import DATA_DIR

DATASETS = ['office31', 'officehome', 'visda', 'domainnet']

DOMAINS = {'office31': ['amazon', 'dslr', 'webcam'], 
           'officehome': ['Art', 'Clipart', 'Product', 'RealWorld'],
           'visda': ['syn', 'real'],
           'domainnet': ['painting', 'real', 'sketch']}

NN = {'office31': [31, 0],
    'officehome': [65, 0],
    'visda': [12, 0],
    'domainnet': [345, 0]}


def save_all_csv(all_headers, all_columns, result_path):
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(result_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(all_columns)


for dataset in DATASETS:
    header_csv = ['label', 'class name',] + DOMAINS[dataset]
    table = np.zeros((NN[dataset][0], len(DOMAINS[dataset])), dtype=int)
    id_domain = 0
    lab2name = dict()
    for domain in DOMAINS[dataset]:
        data = dataset_classes[dataset](DATA_DIR, domain, domain, NN[dataset][0], NN[dataset][1])
        freq = Counter(data.train_labels)
        lab2name.update(data.lab2cname)
        for label in freq:
            table[label, id_domain] = freq[label]
        id_domain += 1
    
    data_csv = []
    for label in range(NN[dataset][0]):
        data_csv.append([label, lab2name[label]] + list(table[label]))

    total_num = table.sum(axis=0)
    data_csv.append(['total', ''] + list(total_num))

    # save_path = os.path.join('latex', 'dataset', f'{dataset}.csv')
    # save_all_csv(header_csv, data_csv, save_path)

    ## for latex use
    data_csv_latex = []

    for item in data_csv:
        new_item = []
        for i in range(len(item)):
            new_item.append(item[i])
            if i == len(item) - 1:
                new_item.append('\\\\ \\hline')
            else:
                new_item.append('&')
        data_csv_latex.append(new_item)
    
    header_csv_latex = []
    for i in range(len(header_csv)):
        header_csv_latex.append(header_csv[i])
        if i == len(header_csv) - 1:
            header_csv_latex.append('\\\\ \\hline')
        else:
            header_csv_latex.append('&')

    save_path_latex = os.path.join('latex', 'dataset', f'{dataset}_latex.csv')
    save_all_csv(header_csv_latex, data_csv_latex, save_path_latex)

