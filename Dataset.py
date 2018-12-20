import pandas as pd


class Dataset(object):
    """ class docs"""
    def __init__(self, path, size='ml-100k'):
        if size == 'ml-1m':
            self.train_ratings = self.load_train_ratings(
                path + 'ml-1m.train.rating')
            self.test_ratings = self.load_train_ratings(
                path + 'ml-1m.test.rating')
        elif size == 'ml-100k':
            self.train_ratings = self.load_train_ratings(
                path + 'ml-100k.train.rating')
            self.test_ratings = self.load_train_ratings(
                path + 'ml-100k.test.rating')
#            self.

    def load_train_ratings(self, path):
        train_ratings = pd.read_csv(
            path,
            header=0,
            names=['userId', 'itemId', 'rating'],
            usecols=['userId', 'itemId', 'rating'])
        return train_ratings


def get_train_instances(df):
    """Non genre version"""
    user_input = df['userId'].values
    item_input = df['itemId'].values
    label = df['rating'].values
    return user_input, item_input, label


def get_train_instances_genre(df):
    """With genre version"""
    user_input = df['userId'].values
    item_input = df['itemId'].values
    label = df['rating'].values
    return user_input, item_input, label


if __name__ == '__main__':
    dataset = Dataset('Data/', 'ml-100k')
    x = dataset.train_ratings
    y = dataset.test_ratings
    print('pause')
