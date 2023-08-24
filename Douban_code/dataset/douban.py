import numpy as np
import pandas as pd
import torch.utils.data

class Douban(torch.utils.data.Dataset):
    """
    Douban Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    Reference:
        https://github.com/FengZhu-Joey/GA-DTCDR/tree/main/Data
    """

    def __init__(self, mode, path_base='./dataset/'):
        #mode: train / val / test
        dataset_name = ['douban_music', 'douban_book', 'douban_movie']
        path = path_base + dataset_name[0] + '_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        for i in range(1, len(dataset_name)):
            path = path_base + dataset_name[i] + '_' + mode + '.csv'
            data_on = pd.read_csv(path).to_numpy()[:, :4]
            data = np.concatenate((data, data_on), 0)
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:, 3].astype(np.float32)
        self.field_dims = np.ndarray(shape=(2,), dtype=int)
        self.field_dims[0] = 2718
        self.field_dims[1] = 21909 # music 5567 + book 6777 + movie 9565

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


class DoubanMusic(torch.utils.data.Dataset):

    def __init__(self, mode, path_base='./dataset/'):
        #mode: train / val / test
        dataset_name = ['douban_music']
        path = path_base + dataset_name[0] + '_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:, 3].astype(np.float32)
        self.field_dims = np.ndarray(shape=(2,), dtype=int)
        self.field_dims[0] = 2718
        self.field_dims[1] = 5567 + 6777 + 9565

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

class DoubanBook(torch.utils.data.Dataset):

    def __init__(self, mode, path_base='./dataset/'):
        #mode: train / val / test
        dataset_name = ['douban_book']
        path = path_base + dataset_name[0] + '_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:, 3].astype(np.float32)
        self.field_dims = np.ndarray(shape=(2,), dtype=int)
        self.field_dims[0] = 2718
        self.field_dims[1] = 5567 + 6777 + 9565

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

class DoubanMovie(torch.utils.data):

    def __init__(self, mode, path_base='/Douban/dataset/'):
        #mode: train / val / test
        dataset_name = ['douban_movie']
        path = path_base + dataset_name[0] + '_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:, 3].astype(np.float32)
        self.field_dims = np.ndarray(shape=(2,), dtype=int)
        self.field_dims[0] = 2718
        self.field_dims[1] = 5567 + 6777 + 9565

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
