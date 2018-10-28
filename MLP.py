import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Lambda, Activation
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.models import Model
#from tensorflow.keras.constraints import maxnorm
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate_legacy import evaluate_model
from datasetclass import Dataset
from time import time
import sys
import multiprocessing as mp

#tf.enable_eager_execution()

class Args(object):
    """A simulator of parser in jupyter notebook"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = '100k'
        self.epochs = 100
        self.batch_size = 256
        self.layers = '[64,32,16,8]'
        self.reg_layers = '[0,0,0,0]'
        self.num_neg = 4
        self.lr = 0.001
        self.learner = 'adam'
        self.verbose = 0
        self.out = 1

def init_normal(shape=[0,0.05], seed=None):
    mean, stddev = shape
    return initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)

def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    concat = Concatenate()
    vector = concat([user_latent, item_latent])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    
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

def fit(name_data = '100k'):
    #args = parse_args()
    args = Args()
    path = args.path
    #dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    # Override args
    args.dataset = name_data

    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    result_out_file = 'outputs/%s_MLP_%s_%d.csv' %(args.dataset, args.layers, time())

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
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    

    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))

    # save Hit ratio and ndcg, loss
    output = pd.DataFrame(columns=['hr', 'ndcg'])
    output.loc[0] = [hr, ndcg]

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    # Generate training instances
    user_input, item_input, labels = get_train_instances(train, num_negatives)

    for epoch in range(epochs):
        t1 = time()
        

        # Training        
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                        np.array(labels), # labels 
                        batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
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
    
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))

    return([best_iter, best_hr, best_ndcg])

if __name__ == '__main__':
    fit('100k')
