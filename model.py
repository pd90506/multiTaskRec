#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:41:07 2018

@author: cid
"""

import torch
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
        self.model = MLP(config)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.MSELoss(reduction='sum')
        if self.config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        #super(MLPEngine, self).__init__(config)
        print(self.model)
        
    def train_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        num_ratings = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor), 'user and item input should be torch.LongTensor'
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            
            if self.config['use_cuda'] is True:
                user = user.cuda()
                item = item.cuda()
                rating = rating.cuda()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
            num_batch_ratings = len(rating)
            num_ratings += num_batch_ratings
            if batch_id % 100 == 0:
                print('[Training Epoch {}] Batch {}, Loss {:3f}'.format(epoch_id, batch_id, loss / num_batch_ratings))
        ave_loss = total_loss / num_ratings 
        print('The total loss for the epoch #{} is {:5.2f}'.format(epoch_id, total_loss))
        print('The average loss for epoch #{} is {:3f}'.format(epoch_id, ave_loss))
        
    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        ratings = ratings.view(-1, 1)
        #print(ratings_pred, ratings)
        loss = self.crit(ratings_pred, ratings)
        loss.backward()
        self.opt.step()
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
    
    
    
    