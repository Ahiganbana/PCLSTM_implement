import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class Mydataset(Dataset):
    def __init__(self, data_dir=None):
        super(Mydataset, self).__init__()
        self.data_dir = data_dir
        # dir_list = self.getDirLists(data_dir)
        self.all_dataset, self.all_dataset_label = self.load_dataset(data_dir)
    
    def load_dataset(self, train_path):
        dataset = []
        labels = []

        for csv_file in os.listdir(os.path.join(train_path, "awacs")):
            file = pd.read_csv(os.path.join(train_path, "awacs", csv_file), encoding='gb2312', header=None)
            data = file.iloc[:, 1:-1].transpose().values.tolist()
            label = file.iloc[1, -1]
            dataset.append(data)
            labels.append(label)

        for csv_file in os.listdir(os.path.join(train_path, "bomber")):
            file = pd.read_csv(os.path.join(train_path, "bomber", csv_file), encoding='gb2312', header=None)
            data = file.iloc[:, 1:-1].transpose().values.tolist()
            # data = np.array(data)
            # print(data.shape)
            label = file.iloc[1, -1]
            dataset.append(data)
            labels.append(label)

        for csv_file in os.listdir(os.path.join(train_path, "ewa")):
            file = pd.read_csv(os.path.join(train_path, "ewa", csv_file), encoding='gb2312', header=None)
            data = file.iloc[:, 1:-1].transpose().values.tolist()
            label = file.iloc[1, -1]
            dataset.append(data)
            labels.append(label)

        for csv_file in os.listdir(os.path.join(train_path, "fighter")):
            file = pd.read_csv(os.path.join(train_path, "fighter", csv_file), encoding='gb2312', header=None)
            data = file.iloc[:, 1:-1].transpose().values.tolist()
            label = file.iloc[1, -1]
            dataset.append(data)
            labels.append(label)

        dataset = np.array(dataset)
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=1)

        return dataset, labels
    
    def __getitem__(self, idx):
        data = self.all_dataset[idx]
        label = self.all_dataset_label[idx]
        data = torch.FloatTensor(data)
        label = torch.tensor(int(label))
        return data, label

    def __len__(self):
        return len(self.all_dataset)