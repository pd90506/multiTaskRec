import GMF
import pandas as pd

if __name__ == '__main__':
    NUM_ITER = 10
    result_out_file = 'outputs/best_GMF-8.csv'
    for i in range(NUM_ITER):
        best_case = GMF.fit()
        output = pd.DataFrame(columns=['best_iter', 'best_hr', 'best_ndcg'])
        output.loc[i] = best_case
        output.to_csv(result_out_file)