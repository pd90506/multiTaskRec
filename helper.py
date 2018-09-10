#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:27:17 2018

@author: cid
"""
import torch
import numpy as np
import random

random.seed(0)

# define a class for manipulating thresholds
class Thresholds():
    def __init__(self, num_thresholds):
        self.step = 0.1
        self.num_thresholds = num_thresholds
        th = np.random.rand(self.num_thresholds)
        th.sort()
        self.thresholds = torch.tensor(th, requires_grad=True)
        self.validate()
        self._err_msg()
        
    def _verify(self):
        levels = len(self.thresholds)
        for i in range(1, levels-2): # skip two inf thresholds
            if self.thresholds[i].item() >= self.thresholds[i+1].item():
                return False
        return True
    
    def _err_msg(self):
        if not self._verify():
            raise Exception("Thresholds validation not pass!")
            
    def validate(self):
        #self.thresholds[0] = self.thresholds[1].item()
        #self.thresholds[self.num_thresholds-1] = self.thresholds[self.num_thresholds-2].item()
        self.thresholds[0] = 0
        self.thresholds[self.num_thresholds-1] = 1
                
#    def validate_v2(self):
#        levels = len(self.thresholds)
#        for i in range(0, levels-1):
#            if self.thresholds[i].item() > self.thresholds[i+1].item():
#                self.thresholds[i+1] = self.thresholds[i] + self.step
    
#    def alter_threshold_v2(self, idx, new_value):
#        self._err_msg()
#        #temp = self.thresholds[idx]
#        self.thresholds[idx] = new_value
#        if not self._verify():
#            if idx == 0:
#                self.thresholds[idx] = self.thresholds[idx + 1] - self.step
#            elif idx == (self.num_thresholds - 1):
#                self.thresholds[idx] = self.thresholds[idx -1] + self.step
#            elif (idx > 0) & (idx < self.num_thresholds - 1):
#                if self.thresholds[idx] > self.thresholds[idx + 1]:
#                    self.thresholds[idx] = self.thresholds[idx + 1] - self.step
#                else:
#                    self.thresholds[idx] = self.thresholds[idx -1] + self.step
#            else:
#                raise Exception("The index is out of bound!")
    
    def alter_threshold(self, idx, new_value):
        self._err_msg()
        temp = self.thresholds[idx].item()
        self.thresholds[idx] = new_value
        if not self._verify():
            self.thresholds[idx] = temp
        self.validate()
        
    def get_threshold(self, idx):
        return self.thresholds[idx].item()        
            
            

    
    
if __name__ == '__main__':
    a = Thresholds(10)
    print(a.thresholds)
    b = a.thresholds[1].item() 
    a.alter_threshold(0, b + 1)
    #a.validate()
    print(a.thresholds)
    a.alter_threshold(0, b - 1)
    print(a.thresholds)







# initiate threshold
def init_thresholds(num_thresholds = 10):
    th = np.random.randn(num_thresholds)
    th.sort()
    thresholds = torch.tensor(th, requires_grad=True)
    return thresholds