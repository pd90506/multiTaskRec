import pandas as pd
import numpy as np

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
    ml_rating = pd.merge(ml_rating, mv_df, on=['mid'], how='left')
    ml_rating = ml_rating[['uid', 'mid', 'rating', 'genre_string']]

    # Reindex 
    user_id = ml_rating[['uid']].drop_duplicates()
    user_id['userId'] = np.arange(len(user_id))
    ml_rating = pd.merge(ml_rating, user_id, on=['uid'], how='left')
    
    item_id = ml_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml_rating = pd.merge(ml_rating, item_id, on=['mid'], how='left')
    ml_rating['rating'] = ml_rating['rating'] # astype(int)

    ml_rating['genre'] = ml_rating['genre_string'].apply(genre_to_int_list)
    ml_rating = ml_rating[['userId', 'itemId', 'rating', 'genre']]
    
    # Data prepared
    print('Range of userId is [{}, {}]'.format(ml_rating.userId.min(), \
          ml_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml_rating.itemId.min(), \
          ml_rating.itemId.max()))
    return(ml_rating)

if __name__ == "__main__":
    y = loadMLData('movielens/ratings.csv', 'movielens/movies.csv')
    new_file_name = 'movielens/genre_set.csv'
    y.to_csv(new_file_name, index=False)
    print("New dataset exported to /{}".format(new_file_name))


