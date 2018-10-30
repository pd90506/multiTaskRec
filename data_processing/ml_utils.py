import pandas as pd
import numpy as np
import random 
import os
from os.path import dirname, abspath
from sklearn.model_selection import train_test_split

def genre_to_int_list(genre_string):
    """
    Convert the list of genre names to a list of integer codes
    Args: 
        genre_string: a string of genres names.
    """
    GENRES = ( 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',\
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',\
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',\
    'Western')
    GENRES_LC = tuple((x.lower() for x in GENRES))
    # convert to lower case
    genre_string_lc = genre_string.lower()
    genre_list = []
    for idx in range(len(GENRES_LC)):
        if GENRES_LC[idx] in genre_string_lc:
            genre_list.append(idx)
    if len(genre_list) == 0:
        genre_list.append(-1)
    return genre_list

def genre_to_single_int(genre_string):
    """
    Convert the list of genre names to a randomly chosen integer code
    Args: 
        genre_string: a string of genres names.
    """
    genre_list = genre_to_int_list(genre_string)
    genre_code = random.choice(genre_list)
    return genre_code

def loadMLData(file_dir, movie_dir):
    """
    Args:
        file_dir: the directory of the data file
        movie_dir: the directory of the movie title genre data file
        
    Load the MovieLens dataset, need to be a csv file
    """

    # read data from file and combine by merging, 
    # select interested columns 
    ml_rating = pd.read_csv(file_dir, header=0, \
                            names=['uid', 'mid', 'rating', 'timestamp'])
    mv_df = pd.read_csv(movie_dir, header=0, \
                            names=['mid', 'title', 'genre_string'])
    mv_df['genre'] = mv_df['genre_string'].apply(genre_to_single_int) # choose which kind of genre to output
    ml_rating = pd.merge(ml_rating, mv_df, on=['mid'], how='left')
    ml_rating = ml_rating.dropna()
    ml_rating = ml_rating[['uid', 'mid', 'rating', 'genre', 'timestamp']]

    # Reindex 
    item_id = ml_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml_rating = pd.merge(ml_rating, item_id, on=['mid'], how='left')
    ml_rating = ml_rating.dropna()

    mv_df_new = pd.merge(mv_df, item_id, on=['mid'], how='left')
    mv_df_new = mv_df_new.dropna()
    mv_df_new = mv_df_new[['itemId', 'genre']].astype(int)

    user_id = ml_rating[['uid']].drop_duplicates()
    user_id['userId'] = np.arange(len(user_id))
    ml_rating = pd.merge(ml_rating, user_id, on=['uid'], how='left')
    
    
    #ml_rating['rating'] = ml_rating['rating'] # astype(int)

    ml_rating = ml_rating[['userId', 'itemId', 'rating', 'genre', 'timestamp']]
    
    # Data prepared
    print('Range of userId is [{}, {}]'.format(ml_rating.userId.min(), \
          ml_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml_rating.itemId.min(), \
          ml_rating.itemId.max()))
    return(ml_rating, mv_df_new)

def sample_negative(ratings):
    """return all negative items & 100 sampled negative items"""
    ## user_pool = set(ratings['userId'].unique())
    item_pool = set(ratings['itemId'].unique())

    interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
        columns={'itemId': 'interacted_items'})
    interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
    interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
    return interact_status[['userId', 'negative_samples']]

def split_train_test(ratings):
    """return training set and test set by loo"""
    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    test = ratings[ratings['rank_latest'] == 1]
    train = ratings[ratings['rank_latest'] > 1]
    assert train['userId'].nunique() == test['userId'].nunique()
    return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]
                

def load_ml(file_dir, movie_dir, genre_dir, train_dir, test_dir, neg_dir):
    """ load movielens dataset
        Args:
            file_dir: rating csv file
            movie_dir: movie info csv file
            genre_dir: output genre csv file
            train_dir: output train csv file
            test_dir: output test csv file
            neg_dir: output negtive samples csv file
    """
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

def load_100k():
    """ load the 100k movielens dataset """
    # get the abspath of parent directory
    p_dir = dirname(dirname(abspath(__file__)))
    file_dir = p_dir + '/movielens/100k-ratings.csv'
    movie_dir = p_dir + '/movielens/100k-movies.csv'
    genre_dir = p_dir + '/Data/ml-100k.genre'
    train_dir = p_dir + '/Data/ml-100k.train.rating'
    test_dir = p_dir + '/Data/ml-100k.test.rating'
    neg_dir = p_dir + '/Data/ml-100k.test.negative'
    load_ml(file_dir, movie_dir, genre_dir, train_dir, test_dir, neg_dir)
    print("100k movielens pre-processing success!")

def load_1m():
    """ load the 1m movielens dataset """
    # get the abspath of parent directory
    p_dir = dirname(dirname(abspath(__file__)))
    file_dir = p_dir + '/movielens/1m-ratings.csv'
    movie_dir = p_dir + '/movielens/1m-movies.csv'
    genre_dir = p_dir + '/Data/ml-1m.genre'
    train_dir = p_dir + '/Data/ml-1m.train.rating'
    test_dir = p_dir + '/Data/ml-1m.test.rating'
    neg_dir = p_dir + '/Data/ml-1m.test.negative'
    load_ml(file_dir, movie_dir, genre_dir, train_dir, test_dir, neg_dir)
    print("1m movielens pre-processing success!")    

if __name__ == "__main__":
    #load_100k()
    
    load_1m()



