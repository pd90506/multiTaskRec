import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    output = pd.read_csv('outputs/eval_result.csv')
    plt.plot(output['ndcg'], label='mt')
    plt.xlabel('epochs')
    plt.ylabel('ndcg')
    plt.title("NDCG plot")
    plt.show()