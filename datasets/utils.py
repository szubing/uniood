import os
import torch
import torchvision
from torchvision.datasets.folder import default_loader
from datasets import dataset_classes
from tools.utils import load_json


class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)


class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        img = self.transform(default_loader(item['impath']))

        output = {
            "img": img,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
            "idx": idx
        }

        return output
    

class FeatureWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, feature_path):
        self.data_source = data_source
        self.features  = torch.load(feature_path)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        feature = self.features[item['impath']]

        output = {
            "img": feature,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
            "idx": idx
        }

        return output
