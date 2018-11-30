from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd
import numpy as np
import random 
import os
from os.path import dirname, abspath
from sklearn.model_selection import train_test_split

def loadMLData(file_dir, train_dir, test_dir, sep):
    """
    Args:
        file_dir: the directory of the data file
        movie_dir: the directory of the movie title genre data file
        
    Load the MovieLens dataset, need to be a csv file
    """

    # read data from file and combine by merging, 
    # select interested columns 
    ml_rating = pd.read_csv(file_dir, header=None, sep=sep,names=['uid', 'mid', 'rating', 'timestamp'])

    # Reindex 
    item_id = ml_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml_rating = pd.merge(ml_rating, item_id, on=['mid'], how='left')
    ml_rating = ml_rating.dropna()

    user_id = ml_rating[['uid']].drop_duplicates()
    user_id['userId'] = np.arange(len(user_id))
    ml_rating = pd.merge(ml_rating, user_id, on=['uid'], how='left')


    ml_rating = ml_rating[['userId', 'itemId', 'rating']]
    
    # Data prepared
    print('Range of userId is [{}, {}]'.format(ml_rating.userId.min(), ml_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml_rating.itemId.min(), ml_rating.itemId.max()))
    
    train, test = train_test_split(ml_rating, test_size=0.2, random_state=1, shuffle=True)
    train.to_csv(train_dir, index=False)
    test.to_csv(test_dir, index=False)
    return(ml_rating)

def create100k():
    p_dir = dirname(dirname(abspath(__file__)))
    file_dir = p_dir + '/movielens/100k-ratings.csv'
    train_dir = p_dir + '/Data/ml-100k.train.rating'
    test_dir = p_dir + '/Data/ml-100k.test.rating'
    loadMLData(file_dir, train_dir, test_dir, sep='\t')

def create1m():
    p_dir = dirname(dirname(abspath(__file__)))
    file_dir = p_dir + '/movielens/1m-ratings.csv'
    train_dir = p_dir + '/Data/ml-1m.train.rating'
    test_dir = p_dir + '/Data/ml-1m.test.rating'
    loadMLData(file_dir, train_dir, test_dir, sep=',')   

if __name__ == "__main__":
    create1m()
