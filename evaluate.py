'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
import pandas as pd
from time import time
#from MGMF import get_model
from datasetclass import Dataset
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, genreList ,K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _genreList
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _genreList = genreList
        
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx, row in testRatings.iterrows():
        (hr,ndcg) = eval_one_rating(idx, row)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx, row):
    rating = row
    items = eval(_testNegatives[idx])
    u = rating['userId']
    gtItem = rating['itemId']
    items.append(int(gtItem))
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    genre = item_to_onehot_genre(items)
    predictions = _model.predict([users, np.array(items), genre], batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def item_to_onehot_genre(items):
    genreList = _genreList
    item_genres = []
    for item in items:
        item_genres.append(genreList[item])
    num_task = 19
    num_items = len(items)
    a = np.zeros((num_items, num_task), int)
    b = np.array(item_genres)
    a[np.arange(num_items), b] = 1
    return a



def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

# if __name__ == '__main__':
#     model = get_model(671, 9125, 8, 19, [0,0])
#     print(model.summary())
#     dataset = Dataset('Data/')
#     train, testRatings, testNegatives, genreList = dataset.train_ratings, dataset.test_ratings, dataset.negatives, dataset.genre
#     (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, genreList, 10, 1)
#     hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
#     print(hr)
#     print(ndcg)