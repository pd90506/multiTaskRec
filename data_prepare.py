import pandas as pd 
import numpy as np

# Load data
ml_dir = 'movielens/ratings.csv'
ml_rating = pd.read_csv(ml_dir, header=0, 
                            names=['uid', 'mid', 'rating', 'timestamp'])

# Reindex 
user_id = ml_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml_rating = pd.merge(ml_rating, user_id, on=['uid'], how='left')

item_id = ml_rating[['mid']].drop_duplicates().reindex()
item_id['itemId'] = np.arange(len(item_id))
ml_rating = pd.merge(ml_rating, item_id, on=['mid'], how='left')
ml_rating['rating'] = ml_rating['rating'].astype(int)

ml_rating = ml_rating[['userId', 'itemId', 'rating']]
# Data prepared
print('Range of userId is [{}, {}]'.format(ml_rating.userId.min(), ml_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml_rating.itemId.min(), ml_rating.itemId.max()))

def loadMLData(file_dir):
    ml_dir = file_dir
    ml_rating = pd.read_csv(ml_dir, header=0, 
                            names=['uid', 'mid', 'rating', 'timestamp'])
    
    # Reindex 
    user_id = ml_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml_rating = pd.merge(ml_rating, user_id, on=['uid'], how='left')
    
    item_id = ml_rating[['mid']].drop_duplicates().reindex()
    item_id['itemId'] = np.arange(len(item_id))
    ml_rating = pd.merge(ml_rating, item_id, on=['mid'], how='left')
    ml_rating['rating'] = ml_rating['rating'].astype(int)
    
    ml_rating = ml_rating[['userId', 'itemId', 'rating']]
    # Data prepared
    print('Range of userId is [{}, {}]'.format(ml_rating.userId.min(), ml_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml_rating.itemId.min(), ml_rating.itemId.max()))

print("hello")