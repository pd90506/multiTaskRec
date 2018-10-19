import pandas as pd 
import numpy as np 
from data_utils import loadMLData, split_train_test, sample_negative
from datasetclass import Dataset

def load_ml(file_dir='movielens/1m-ratings.csv', movie_dir='movielens/1m-movies.csv',\
genre_dir='Data/ml-1m.genre', train_dir='Data/ml-1m.train.rating',\
test_dir='Data/ml-1m.test.rating', neg_dir='Data/ml-1m.test.negative'):
    """ load movielens dataset """
    y, mv = loadMLData(file_dir, movie_dir)
    movie_genre_name = genre_dir
    train_file_name = train_dir
    test_file_name = test_dir
    negative_file_name = neg_dir
    # split train test
    train, test = split_train_test(y)
    # write to csv
    train.to_csv(train_file_name, index=False)
    test.to_csv(test_file_name, index=False)
    mv = mv.sort_values('itemId')
    mv.to_csv(movie_genre_name, index=False)
    negatives = sample_negative(y)
    negatives.to_csv(negative_file_name, index=False)

def load_ml_100k():
    """ load movielens 100k dataset """
    load_ml(file_dir='movielens/100k-ratings.csv', movie_dir='movielens/100k-movies.csv',\
    genre_dir='Data/ml-100k.genre', train_dir='Data/ml-100k.train.rating', \
    test_dir='Data/ml-100k.test.rating', neg_dir='Data/ml-100k.test.negative')
    
def load_ml_1m():
    """ load movielens 1m dataset """
    load_ml()

if __name__ == '__main__':
    load_ml_1m()

