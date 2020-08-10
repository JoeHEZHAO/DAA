import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaussianGridDataset import gaussianGridDataset, gaussianGrid1dDataset

def getDataset(dataset_config):
    dataset_name = dataset_config['name']
    if dataset_name == "gaussian_grid":
        return gaussianGridDataset(dataset_config['n'], dataset_config['n_data'], dataset_config['sig'])
    else:
        raise ValueError("no such dataset called "+dataset_name)

def get1dDataset(dataset_config):
    dataset_name = dataset_config['name']
    if dataset_name == "gaussian_grid":
        return gaussianGrid1dDataset(dataset_config['n'], dataset_config['n_data'], dataset_config['sig'])
    else:
        raise ValueError("no such dataset called "+dataset_name)
