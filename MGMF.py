import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation
from tensorflow.keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten, Dot
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from datasetclass import Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math

class Args(object):
    """A simulator of parser in jupyter notebook"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = '1m'
        self.epochs = 100
        self.batch_size = 256
        self.num_factors = 8
        self.regs = '[0,0]'
        self.num_neg = 4
        self.lr = 0.001
        self.learner = 'adam'
        self.verbose = 1
        self.out = 1
        self.num_tasks = 19

def init_normal(shape=[0.0, 0.05], seed=None):
    mean, stddev = shape
    return initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)

def get_model(num_users, num_items, latent_dim, num_tasks, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    task_input = Input(shape=(num_tasks,), dtype='float', name = 'task_input') # one-hot task input

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(regs[1]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    # Element-wise product of user and item embeddings 
    predict_vector = Multiply()([user_latent, item_latent])
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    #predict_vector = Dense(16, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'fully-connected')(predict_vector)
    predictions = Dense(num_tasks, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'predictions')(predict_vector)
    task_prediction = Dot(1)([predictions, task_input])
    
    model = Model(inputs=[user_input, item_input, task_input], 
                outputs=[task_prediction])

    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_items = len(train['itemId'].unique())
    for _, row in train.iterrows():
        # positive instance
        u = row['userId']
        i = row['itemId']
        user_input.append(int(u))
        item_input.append(int(i))
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            # while ((u,j) in train.keys()):
            #     j = np.random.randint(num_items)
            user_input.append(int(u))
            item_input.append(int(j))
            labels.append(0)
    return user_input, item_input, labels

def item_to_onehot_genre(items, genreList):
    #genreList = _genreList
    item_genres = []
    for item in items:
        item_genres.append(genreList[item])
    num_task = 19
    num_items = len(items)
    a = np.zeros((num_items, num_task), int)
    b = np.array(item_genres)
    a[np.arange(num_items), b] = 1
    return a

def fit():
    args = Args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    num_tasks = args.num_tasks
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("MGMF arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_MGMF_%d_%d.h5' %(args.dataset, num_factors, time())
    result_out_file = 'outputs/%s_MGMF_%d_%d.csv' %(args.dataset, num_factors, time())
    # Loading data
    t1 = time()

    if args.dataset=='1m':
        num_users = 6040
        num_items = 3900
    elif args.dataset=='100k':
        num_users = 671
        num_items = 9125
    else:
        raise Exception('wrong dataset size!!!')   

    dataset = Dataset(args.path, args.dataset)
    train, testRatings, testNegatives, genreList = dataset.train_ratings, dataset.test_ratings, dataset.negatives, dataset.genre

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
        %(time()-t1, num_users, num_items, train.shape[0], testRatings.shape[0]))

    # Build model
    model = get_model(num_users, num_items, num_factors, num_tasks ,regs)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    # test initial outputs
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, genreList, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))\

    # TensorBoard Callbacks
    #tbCallBack = keras.callbacks.TensorBoard(log_dir='logs')
    # save Hit ratio and ndcg, loss
    output = pd.DataFrame(columns=['hr', 'ndcg'])
    output.loc[0] = [hr, ndcg]

    # Generate training instances
    user_input, item_input, labels = get_train_instances(train, num_negatives)
    genre_input = item_to_onehot_genre(item_input, genreList)
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input), genre_input], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, genreList, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            output.loc[epoch+1] = [hr, ndcg]
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    output.to_csv(result_out_file)
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))

if __name__ == '__main__':
    fit()
    
  