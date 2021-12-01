import re
import os
import random
import torch
from torch.utils.data import dataset
import pandas as pd
import numpy as np

seed_num = 233
torch.manual_seed(seed_num)
random.seed(seed_num)


class Mydataset(dataset.Dataset):
    def __init__(self, data_dir=None):
        super(Mydataset, self).__init__()
        self.data_dir = data_dir
        dir_list = self.getDirLists(data_dir)
        self.all_dataset, self.all_dataset_label = self.load_dataset(dir_list)

    
    def getDirLists(self, data_dir):
        pathDir = os.listdir(data_dir)
        childs = []
        for all_dir in pathDir:
            child = os.path.join('%s\\%s' % (data_dir, all_dir))
            childs.append(child)
        return childs
    
    def load_dataset(self, dir_list):
        all_dataset = []
        all_dataset_label = []
        for dir_item in dir_list:
            for _, _, files in os.walk(dir_item):
                for file in files:
                    if os.path.splitext(file)[1] == '.csv':
                        file_full_path = os.path.join(dir_item, file)
                        data_row = pd.read_csv(file_full_path, encoding='gb2312', header=None, index_col=None)
                        all_dataset.append(data_row.iloc[:,1:-1].transpose().values.tolist())
                        all_dataset_label.append(data_row.iloc[1,-1])
            # all_dataset.append(dir_list_item)
            # all_dataset_label.append(dir_list_label)
        return all_dataset, all_dataset_label
    
    def __getitem__(self, idx):
        data = self.all_dataset[idx]
        label = self.all_dataset_label[idx]
        data = torch.tensor(data)
        label = torch.tensor(label)
        return data, label

    def __len__(self):
        return len(self.all_dataset)


