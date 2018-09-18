#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:44:05 2018
@author: cid
"""
import pandas as pd 
import numpy as np
import torch
import random
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset



def loadMLData(file_dir):
    """
    Args:
        file_dir: the directory of the data file
        
    Load the MovieLens dataset, need to be a csv file
    """
    mv_dir = 'data/movielens/movies.csv'
    ml_dir = (file_dir, mv_dir)
    
    ml_rating = pd.read_csv(ml_dir[0], header=0, 
                            names=['uid', 'mid', 'rating', 'timestamp'])
    mv_info = pd.read_csv(ml_dir[1], header=0,
                            names=['mid', 'title', 'genre'])
    ml_rating = ml_rating.merge(mv_info, on=['mid'], how='left')
    
    # Reindex 
    user_id = ml_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml_rating = pd.merge(ml_rating, user_id, on=['uid'], how='left')
    
    item_id = ml_rating[['mid']].drop_duplicates().reindex()
    item_id['itemId'] = np.arange(len(item_id))
    ml_rating = pd.merge(ml_rating, item_id, on=['mid'], how='left')
    ml_rating['rating'] = ml_rating['rating'] # astype(int)
    
    ml_rating = ml_rating[['userId', 'itemId', 'rating', 'timestamp', 'genre']]
    # Data prepared
    print('Range of userId is [{}, {}]'.format(ml_rating.userId.min(), 
          ml_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml_rating.itemId.min(), 
          ml_rating.itemId.max()))
    return(ml_rating)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for
            <user, item> pair
        """
        if len(user_tensor) == len(item_tensor) & \
        len(user_tensor) == len(target_tensor):
            
            self.user_tensor = user_tensor
            self.item_tensor = item_tensor
            self.target_tensor = target_tensor
        else:
            raise Exception('user, item and target length should be the same.')
        

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], \
                self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
    
    
    
class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.normalize_ratings = self._normalize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.n_users = len(self.user_pool)
        self.item_pool = set(self.ratings['itemId'].unique())
        self.n_items = len(self.item_pool)

        self.train_ratings, self.test_ratings = self._split_loo(self.normalize_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating]"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]


    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users = self.train_ratings['userId'].values
        items = self.train_ratings['itemId'].values
        ratings = self.train_ratings['rating'].values
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_users = self.test_ratings['userId'].values
        test_items = self.test_ratings['itemId'].values
        test_ratings = self.test_ratings['rating'].values
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), \
                torch.FloatTensor(test_ratings)]

    
if __name__ == '__main__':
    y = loadMLData('data/movielens/ratings.csv')
    sample_generator = SampleGenerator(ratings=y)
    
    
    train_loader = sample_generator.instance_a_train_loader(4, 128)