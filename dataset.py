"""Dataset module to prepare data for the learning pipeline"""
# Copyright (C) 2020 Amir Alansary <amiralansary@gmail.com>

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


###############################################################################
def count_samples_per_class(labels):
    """
    Count number of samples per class
    :param labels:
    :return: vector of size(labels) containing each class frequency
    """
    return torch.FloatTensor([len(np.where(labels == t)[0]) for t in np.unique(labels)])


def assign_sample_weight(labels, weights):
    """
    Assign weight for each sample
    :param labels:
    :param weights:
    :return:
    """
    return torch.FloatTensor([weights[t] for t in labels])


def compute_class_weight(n_samples, n_classes, class_bincount):
    """
    Estimate class weights for unbalanced datasets.
    Class weights are calculated by: n_samples / (n_classes * class_sample_count)
    :param n_samples:
    :param n_classes:
    :param class_bincount:
    :return:
    """
    return torch.FloatTensor(n_samples / (n_classes * class_bincount))


###############################################################################
###############################################################################

class HeartDiseaseDataset(Dataset):
    """A dataset class to retrieve samples of paired images and labels"""

    def __init__(self, csv, shuffle=None, label_names=None):
        """
        Args:
            csv (string): Path to the csv file with data
            shuffle (callable, optional): Shuffle list of files
        """
        super().__init__()
        # self.transform = transform
        self.csv_file = pd.read_csv(csv)
        self.label_names = label_names

        labels = self.csv_file['num']

        self.class_sample_count = count_samples_per_class(labels)
        self.class_probability = self.class_sample_count / len(labels)
        self.sample_weights = assign_sample_weight(labels, 1. / self.class_sample_count)
        self.class_weights = compute_class_weight(n_samples=self.__len__(),
                                                  n_classes=len(self.class_sample_count),
                                                  class_bincount=self.class_sample_count)

        if shuffle:
            self.csv_file = self.csv_file.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = torch.tensor(self.csv_file.iloc[:, :13].values[idx]).float()
        target = torch.tensor(self.csv_file['num'].values[idx])

        return {'features': features, 'target': target}
