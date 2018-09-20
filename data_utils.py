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

def num_genre(df, genre_code):
    """
    Check how many movies are in this genre
    """
    genre_list = list(df.genre)
    count = 0
    for genres in genre_list:
        for genre in genres:
            if genre == genre_code:
                count += 1
    return count
    

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

def expand_genre(df):
    """
    Expand samples based on the number of genres a sample belongs to
    Args:
        df: a dataframe contains a column 'genre'
    """
    column_names = df.columns.values
    new_df = pd.DataFrame(columns=column_names)
    count = 0
    for index, row in df.iterrows():
        genres = row['genre']
        for genre in genres:
            new_sample = {}
            for name in column_names:
                if name == 'genre':
                    new_sample[name] = genre
                else:
                    new_sample[name] = row[name]
            new_df = new_df.append(new_sample, ignore_index=True)
            if count % 10000 == 9999:
                print('current progress is {}/262343'.format(count))
                #return new_df #!!! test a small dataframe
            count += 1
    return new_df

def convert_datatype(df):
    df['userId'] = df['userId'].astype('int64')
    df['itemId'] = df['itemId'].astype('int64')
    df['genre'] = df['genre'].astype('int64')
    return df


if __name__ == "__main__":
    y = loadMLData('movielens/ratings.csv', 'movielens/movies.csv')
    print(y.head())
    #print(y.head())
    # count = 0
    # for i in range(18):
    #     print(num_genre(y,i))
    #     count += num_genre(y,i)
    # print('total count is ', count)
    new_file_dir = 'expanded_dataset.csv'
    new_df = expand_genre(y)
    convert_datatype(new_df)
    new_df.to_csv(new_file_dir, index=False)
    print(new_df.head())
