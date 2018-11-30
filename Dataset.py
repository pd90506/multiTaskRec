import pandas as pd

class Dataset(object):
    """ class docs"""
    def __init__(self, path, size='ml-100k'):
        if size == 'ml-1m':
            self.train_ratings = self.load_train_ratings(path + 'ml-1m.train.rating')
            self.test_ratings = self.load_train_ratings(path + 'ml-1m.test.rating')
        elif size =='ml-100k':
            self.train_ratings = self.load_train_ratings(path + 'ml-100k.train.rating')
            self.test_ratings = self.load_train_ratings(path + 'ml-100k.test.rating')
        
    
    def load_train_ratings(self, path):
        train_ratings = pd.read_csv(path, header=0, names=['userId', 'itemId', 'rating'])
        return train_ratings



if __name__ == '__main__':
    dataset = Dataset('Data/', 'ml-100k')
    x = dataset.train_ratings
    y = dataset.test_ratings
    print('pause')