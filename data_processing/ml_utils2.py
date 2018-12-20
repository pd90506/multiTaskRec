from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd
import numpy as np
import random 
import os
from os.path import dirname, abspath
from sklearn.model_selection import train_test_split

def load_100k():
    # read data from file and combine by merging, 
    # select interested columns 
    rating_dir = 'movielens/100k-ratings.csv'
    train_dir = 'Data/ml-100k.train.rating'
    test_dir = 'Data/ml-100k.test.rating'
    movie_dir ='movielens/100k-movies'
    genre_dir = 'Data/ml-100k.genre'

    ml_rating = pd.read_csv(rating_dir, header=None, sep='\t',names=['uid', 'mid', 'rating', 'timestamp'])

    col_names = ['mid', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western']
    movies = pd.read_csv(movie_dir, sep='|', names=col_names, encoding='latin-1')
    genre_names = ['mid', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western']
    movies = movies[genre_names]

    # merge ml_rating and movies
    ml_rating = pd.merge(ml_rating, movies, on=['mid'], how='left')

    # Reindex 
    item_id = ml_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml_rating = pd.merge(ml_rating, item_id, on=['mid'], how='left')
    ml_rating = ml_rating.dropna()

    user_id = ml_rating[['uid']].drop_duplicates()
    user_id['userId'] = np.arange(len(user_id))
    ml_rating = pd.merge(ml_rating, user_id, on=['uid'], how='left')

    movie_genre = ml_rating[['itemId', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western']]
    ml_rating = ml_rating[['userId', 'itemId', 'rating','unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western']]

    # Data prepared
    print('Range of userId is [{}, {}]'.format(ml_rating.userId.min(), ml_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml_rating.itemId.min(), ml_rating.itemId.max()))

    train, test = train_test_split(ml_rating, test_size=0.2, random_state=1, shuffle=True)
    train.to_csv(train_dir, index=False)
    test.to_csv(test_dir, index=False)
    #movie_genre.to_csv(genre_dir, index=False)

if __name__ == "__main__":
    load_100k()
