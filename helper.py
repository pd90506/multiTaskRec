#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:27:17 2018

@author: cid
"""
import torch
import numpy as np
# initiate threshold
def init_thresholds(num_thresholds = 10):
    th = np.random.randn(num_thresholds)
    th.sort()
    thresholds = torch.tensor(th, requires_grad=True)
    return thresholds
    
if __name__ == '__main__':
    a = init_thresholds(10)
    print(a)
    

class Thresholds():