import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from ml_dataset import load_data

class SampleGenerator(object):
    """Construct training and test dataset for NCF"""
    def __init__(self, dataset):
        """
        args:
            dataset: MLDataset object in ml_dataset.py
        """
        self.dataset = dataset

    def train_loader(self, num_negatives=4, batch_size=128):
        """
        sample a training set with num_negatives negative samples
        args:
            num_negatives: number of negative samples per positive sample
            batch_size: batch size
        """
        random.seed(0)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, \
            shuffle=True, num_workers=0)
        return dataloader
    
    #def test_loader(self, num_negatives=4, batch_size=128):

        

if __name__ == '__main__':
    mldata = load_data()
    generator = SampleGenerator(mldata)
    train_data = generator.train_loader()
    for i_batch, sample_batched in enumerate(train_data):
        print('batch #{}, {} batched samples'.format(i_batch, sample_batched['userId'].size()))

        if i_batch == 3:
            break 