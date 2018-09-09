#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:54:19 2018

@author: cid
"""

#import torch
from model import MLPEngine
from data_helper import loadMLData, SampleGenerator
import matplotlib.pyplot as plt

# set configuration
if __name__ == '__main__':
    
    # Load data using data_helper
    ratings = loadMLData('movielens/ratings.csv')
    sample_generator = SampleGenerator(ratings=ratings)    
    n_users = sample_generator.n_users
    n_items = sample_generator.n_items
     
    
    
    
    config = {'num_epoch': 1,
              'batch_size': 128,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': n_users,
              'num_items': n_items,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'device_id': 0 }
    engine = MLPEngine(config)
    
    for epoch in range(config['num_epoch']):
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])   
        evaluate_data = sample_generator.evaluate_data
             
        
        engine.train_epoch(train_loader, epoch, evaluate_data)
        loss = engine.evaluate(evaluate_data, epoch_id=epoch)
        print('The testing loss for epoch #{} is {:3f}\n'.format(epoch, loss))
    
    batch = engine.train_batch
    train_loss = engine.train_loss
    test_loss = engine.test_loss
    plt.figure()
    plt.plot(batch, train_loss, label='training_loss')
    plt.plot(batch, test_loss, label='test_loss')
    plt.title('Loss during training')
    plt.xlabel('batch')
    plt.ylabel('SSE loss')
    plt.legend()
    plt.show()