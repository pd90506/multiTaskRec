import pandas as pd 

def dat_to_csv(source, target):
    df = pd.read_csv(source, sep='::', header=0)
    df.to_csv(target, index=False)


if __name__ == '__main__':
    #dat_to_csv('movielens/ratings.dat', 'movielens/1m-ratings.csv')
    #dat_to_csv('movielens/movies.dat', 'movielens/1m-movies.csv')