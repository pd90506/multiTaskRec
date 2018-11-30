'''
Created on Aug 9, 2016

Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import pandas as pd

from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation
from tensorflow.keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from datasetclass import Dataset
from evaluate_legacy import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math

class Args(object):
    """A simulator of parser in jupyter notebook"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = '100k'
        self.epochs = 100
        self.batch_size = 2048
        self.num_factors = 8
        self.regs = '[0,0]'
        self.num_neg = 4 
        self.lr = 0.001
        self.learner = 'adam'
        self.verbose = 0
        self.out = 1

def init_normal(shape=[0,1], seed=None):
    mean, stddev = shape
    return initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)

def get_model(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

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
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                outputs=[prediction])

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



def fit(name_data = '100k', batch_size=2048):
    #args = parse_args()
    args = Args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    num_epochs = args.epochs
    #batch_size = args.batch_size
    verbose = args.verbose
    

    # Override args
    args.dataset = name_data
    args.batch_size = batch_size
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("GMF arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' %(args.dataset, num_factors, time())
    result_out_file = 'outputs/%s_GMF_%d_%d.csv' %(args.dataset, num_factors, time())
    
    # Loading data
    t1 = time()
    if args.dataset=='1m':
        num_users = 6040
        num_items = 3706
    elif args.dataset=='100k':
        num_users = 943
        num_items = 1682
    else:
        raise Exception('wrong dataset size!!!')   

    dataset = Dataset(args.path, args.dataset)
    train, testRatings, testNegatives, genreList = dataset.train_ratings, dataset.test_ratings, dataset.negatives, dataset.genre

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
        %(time()-t1, num_users, num_items, train.shape[0], testRatings.shape[0]))
    
    # Build model
    model = get_model(num_users, num_items, num_factors, regs)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    #print(model.summary())
        # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    #mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
    #p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))

    # save Hit ratio and ndcg, loss
    output = pd.DataFrame(columns=['hr', 'ndcg'])
    output.loc[0] = [hr, ndcg]


    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    # Generate training instances
    user_input, item_input, labels = get_train_instances(train, num_negatives)

    for epoch in range(int(num_epochs)):
        t1 = time()

        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %1 == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            output.loc[epoch+1] = [hr, ndcg]
            if ndcg > best_ndcg:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)
    

    output.to_csv(result_out_file)
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best GMF model is saved to %s" %(model_out_file))

    return([best_iter, best_hr, best_ndcg])


if __name__ == '__main__':
    fit('100k')