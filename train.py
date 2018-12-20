import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from Dataset import Dataset
import gmf_model

class Args(object):
    """A simulator of parser in jupyter notebook"""
    def __init__(self):
        self.model = 'gmf'
        self.path = 'Data/'
        self.data = 'ml-100k'
        self.epochs = 50
        self.batch_size = 256
        self.factors = 16
        self.regs = [0.00001,0.00001]
        self.learning_rate = 0.001
        self.loss = root_mean_squared_error
        self.metrics = ['mae', root_mean_squared_error]
        self.learner = 'adam'
        self.verbose = 0
        if self.data == 'ml-1m':
            self.num_users = 6040
            self.num_items = 3706
        elif self.data == 'ml-100k':
            self.num_users = 943
            self.num_items = 1682
        else:
            raise ValueError('No a valid dataset name.')


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_a_model(args):
    if args.model == 'gmf':
        return gmf_model.get_model(num_users=args.num_users, num_items=args.num_items, factor=args.factors, regs=args.regs)

def get_train_instances(df):
    user_input = df['userId'].values
    item_input = df['itemId'].values
    label = df['rating'].values
    return user_input, item_input, label


def fit(args):
    learning_rate = args.learning_rate
    path = args.path  
    batch_size = args.batch_size
    verbose = args.verbose
    loss = args.loss
    metrics=args.metrics
    epochs = args.epochs
    data = args.data

    # get training model
    model = get_a_model(args)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

    dataset = Dataset(path, size=data)
    train, test = dataset.train_ratings, dataset.test_ratings
    train_user, train_item, train_label = get_train_instances(train)
    test_user, test_item, test_label = get_train_instances(test)

    # start training iterations
    # init the best loss and iters
    best_loss, best_mae, best_mse = model.evaluate([test_user, test_item], test_label, batch_size=batch_size, verbose=verbose)
    best_iter = -1
    print('The initial performance, MAE = {}, RMSE = {}'.format(best_mae, best_mse))
    for i in range(0,epochs):
        print('Iteration:{}'.format(i))
        hist = model.fit([np.array(train_user), np.array(train_item)], np.array(train_label), batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        test_loss = model.evaluate([test_user, test_item], test_label, batch_size=batch_size, verbose=verbose)
        print('The test MAE = {}, RMSE = {}'.format(test_loss[1], test_loss[2]))
        # select the best iter based on MAE
        if test_loss[2] < best_mse:
            best_iter = i
            best_loss, best_mae, best_mse = test_loss
    
    print('The best iteration is {}, with MAE = {}, RMSE = {}'.format(best_iter, best_mae, best_mse))
        


if __name__ == "__main__":
    args = Args()
    fit(args)