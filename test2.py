# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:40:09 2018

@author: pd905
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:35:56 2018

@author: Deng Pan (panda)

"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time




class MovieLensDataset(Dataset):
    """
    MovieLens dataset
    """
    def __init__(self, csv_file):
        """
        Args:
            csv_file: the path of the csv file
            
        """
        super(MovieLensDataset, self).__init__()
        self.movielens = pd.read_csv(csv_file)
        self.shape = self.movielens.shape
        self.n_users = len(self.movielens.iloc[:, 0].unique())
        self.n_items = len(self.movielens.iloc[:, 1].unique())
        

    def __len__(self):
        return self.movielens.shape[0]
    
    def __getitem__(self, idx):
        user = self.movielens.iloc[idx, 0]
        item = self.movielens.iloc[idx, 1]
        rating = self.movielens.iloc[idx, 2]
        sample = {'input' : np.array([user,item]),
                  'rating': int(rating)}
        
    
        return sample
    


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        # each epoch process
        for sample in dataloader:
            inputs = sample.get('input')
            ratings = sample.get('rating')
            # move to GPU
            inputs = inputs.to(device)
            ratings = ratings.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward and track
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, ratings)
            
            loss.backward()
            optimizer.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == ratings.data)
            
    time_elapsed = time.time() - since       
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))        


class Net(nn.Module):
    
    def __init__(self, n_users, n_items, n_factors=100):
        super(Net, self).__init__()
        # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                              sparse=True)
        # create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                              sparse=True)
        
        # the dimension after concatenation
        input_dim = 2 * n_factors
        # define layers
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50,10)
        self.fc3 = nn.Linear(10,5)
        self.softmax1 = nn.Softmax(dim=0)
         
    def forward(self, input_data):
        print("good")
        user = input_data[:,0]
        item = input_data[:,1]
        user_embeded = self.user_factors(user)
        item_embeded = self.item_factors(item)
        # concatenated tensor from user_embeded and item_embeded
        # TO DO: PayAttention
        x = torch.cat((user_embeded, item_embeded), dim=1)
        print(x.shape)
        # feed forward
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax1(x)
        
        return x
    
    
    
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    csv_file = "movielens/ratings.csv"
    movie_dataset = MovieLensDataset(csv_file)
    
    dataloader = DataLoader(movie_dataset, batch_size=4, shuffle=True, 
                            num_workers=0)
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    n_users = movie_dataset.n_users
    n_items = movie_dataset.n_items
    
    model = Net(n_users, n_items, n_factors = 100)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)   

    train_model(model, criterion, optimizer, num_epochs=25) 
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        