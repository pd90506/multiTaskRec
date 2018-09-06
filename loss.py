#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:48:33 2018

@author: cid
"""

#import torch
from torch.nn import MSELoss
from torch.nn import functional as F
class OrdinalMSELoss(MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(OrdinalMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target1, target2):
        loss1 = F.mse_loss(input, target1, reduction=self.reduction)
        loss2 = F.mse_loss(input, target2, reduction=self.reduction)
        return loss1 + loss2
    
    
#class OrdinalMSELoss(MSELoss):
#    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
#        super(OrdinalMSELoss, self).__init__(size_average, reduce, reduction)
#
#    def forward(self, thresholds, input, target):
#        threshold1 = 
#        threshold2 = 
#        loss1 = F.mse_loss(input, target1, reduction=self.reduction)
#        loss2 = F.mse_loss(input, target2, reduction=self.reduction)
#        return loss1 + loss2