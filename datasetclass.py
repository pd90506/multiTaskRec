import pandas as pd

class Dataset(object):
    """ class docs"""
    def __init__(self, path):
        self.train_ratings = self.load_train_ratings(path + 'ml-1m.train.rating')
        self.test_ratings = self.load_train_ratings(path + 'ml-1m.test.rating')
        self.negatives = self.load_negatives(path + 'ml-1m.test.negative')
        self.genre = self.load_genre(path + 'movie.genre.csv')
        #assert self.test_ratings.shape[0] == self.negatives.shape[0]
        
    
    def load_train_ratings(self, path):
        train_ratings = pd.read_csv(path, header=0, names=['userId', 'itemId', 'rating'])
        return train_ratings

    def load_negatives(self, path):
        negatives = pd.read_csv(path, header=0, names=['userId', 'negatives'])
        negativeList = negatives['negatives'].values.tolist()
        return negativeList

    def load_genre(self, path):
        genre = pd.read_csv(path, header=0, names=['itemId', 'genre'])
        genreList = genre['genre'].values.tolist()
        return genreList

if __name__ == '__main__':
    dataset = Dataset('Data/')
    x = dataset.train_ratings
    y = dataset.test_ratings
    z = dataset.negatives
    a = dataset.genre
    print('pause')
