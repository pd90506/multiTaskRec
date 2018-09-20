import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

class MLDataset(Dataset):
    """Wrapper, convert csv into Pytorch Dataset"""
    def __init__(self, file_dir, transform=None):
        """
        args:
            file_dir: the file directory of the extended ML data file
            ransform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(file_dir)
        self.transform = transform
        


    def __getitem__(self, idx):
        """
        note that the returned item is a Dict instance.
        """
        row = self.df.loc[idx]
        new_row = {} 
        new_row['userId'] = row['userId'].astype('int')
        new_row['itemId'] = row['itemId'].astype('int')
        new_row['rating'] = row['rating'].astype('float')
        new_row['genre'] = row['genre'].astype('int')
        sample = new_row
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.df)

# using this function to load data directly
def load_data(file_dir='expanded_dataset.csv'):
    mldata = MLDataset(file_dir)
    return mldata

if __name__ == '__main__':
    mldata = load_data()
    print(mldata[0])