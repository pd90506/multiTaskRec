#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:41:07 2018

@author: cid
"""

import torch
from loss import OrdinalMSELoss
from helper import Thresholds
import numpy as np
#from gmf import GMF
#from engine import Engine
#from utils import use_cuda, resume_checkpoint

def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)

def use_optimizer(network, params):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=params['adam_lr'], weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        
    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class MLPEngine():
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.config = config
        self.num_thresholds = config['num_rating_levels'] + 1  # including -inf and +inf, set to equal to t1 and t10 respectively
        self.thresholds = Thresholds(self.num_thresholds) # define thresholds [0-10] , beware that ratings are [1-10]
        self.model = MLP(config)
        self.opt = use_optimizer(self.model, config)
        self.crit = OrdinalMSELoss(reduction='sum')
        self.train_batch = []
        self.train_loss = []
        self.test_loss = []
        
        if self.config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        #super(MLPEngine, self).__init__(config)
        print(self.model)
        
    def train_epoch(self, train_loader, epoch_id, evaluate_data):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        num_ratings = 0
        update_th = False
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor), 'user and item input should be torch.LongTensor'
            user, item, rating = batch[0], batch[1], batch[2]
            #rating = rating.float() # do not need to be float anymore
            if self.config['use_cuda'] is True:
                user = user.cuda()
                item = item.cuda()
                rating = rating.cuda()
            loss = self.train_single_batch(user, item, rating, update_th)
            update_th = not update_th
            total_loss += loss
            num_batch_ratings = len(rating)
            num_ratings += num_batch_ratings
            if batch_id % 50 == 0:
                self.train_batch.append(batch_id)
                self.train_loss.append(loss.item()/ num_batch_ratings)
                self.evaluate_v2(evaluate_data, epoch_id)
                print('[Training Epoch {}] Batch {}, Loss {:3f}'.format(epoch_id, batch_id, loss / num_batch_ratings))
        ave_loss = total_loss / num_ratings 
        print('The total loss for the epoch #{} is {:5.2f}'.format(epoch_id, total_loss))
        print('The average loss for epoch #{} is {:3f}'.format(epoch_id, ave_loss))
        
    def train_single_batch(self, users, items, ratings, update_th=False):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
                users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
                
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        ratings = ratings.long()
        #print(ratings_pred, ratings)
        th = self.thresholds.thresholds
        th1 = torch.empty(len(ratings))
        th2 = torch.empty(len(ratings))
        for i in range(len(ratings)):
            idx = ratings[i].item()
            th1[i] = th[idx-1].item()
            th2[i] = th[idx].item()
        th1 = th1.view(-1,1)
        th2 = th2.view(-1,1)
        loss = self.crit(ratings_pred, th1, th2)
        if update_th:
            #todo
            self.update_thresholds(ratings_pred, ratings)
        else:
            loss.backward()
            self.opt.step()
        #self.update_th() # implement!
        
        ### if flag: update only thresholds
        
        #if self.config['use_cuda'] is True:
        #    loss = loss.data.cpu().numpy()[0]
        #else:
        #    loss = loss.data.numpy()[0]
        return loss

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        test_users, test_items, ratings = evaluate_data[0], evaluate_data[1], evaluate_data[2]
        #negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()

        test_scores = self.model(test_users, test_items)
        ratings = ratings.view(-1, 1)
        loss = self.crit(test_scores, ratings) / ratings.shape[0]
        
        return loss
    
    def evaluate_v2(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        test_users, test_items, ratings = evaluate_data[0], evaluate_data[1], evaluate_data[2]
        #negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()

        test_scores = self.model(test_users, test_items)
        ratings = ratings.view(-1, 1).int()
        ratings = ratings.numpy()
        evaluated_ratings = []
        th = self.thresholds.thresholds
        for score in test_scores:
            for i in range(self.num_thresholds):
                if th[i].item() >= score.item():
                    evaluated_ratings.append(i)
                    break
        
        acc = sum(1 for x,y in zip(evaluated_ratings,ratings) if x == y) / len(ratings)
        self.test_loss.append(acc)
        
    
    def update_thresholds(self, ratings_pred, ratings, lr=1e-5):
        lr = lr # set learning rate
        ratings_pred = ratings_pred.view(-1)
        ratings = ratings.view(-1)
        th = self.thresholds
        th_temp = np.random.random(self.num_thresholds)
        for i in range(self.num_thresholds):
            th_temp[i] = th.get_threshold(i)
        for i in range(len(ratings)):
            th_idx = ratings[i].item()
            grad1 = th_temp[th_idx] - ratings_pred[i].item()
            grad2 = th_temp[th_idx - 1] - ratings_pred[i].item()
            th_temp[th_idx] -= lr * grad1
            th_temp[th_idx - 1] -= lr * grad2
            th.alter_threshold(th_idx, th_temp[th_idx])
            th_temp[th_idx] = th.get_threshold(th_idx)
            th.alter_threshold(th_idx - 1, th_temp[th_idx - 1])
            th_temp[th_idx-1] = th.get_threshold(th_idx-1)
       # print(th.thresholds)
            
            
            
        
        
    
    
    