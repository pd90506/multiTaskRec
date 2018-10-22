import GMF
import MGMF
import pandas as pd
from data_processing.ml_utils import load_100k


def run_model(model, output_dir, num_iter=10):
    for i in range(num_iter):
        load_100k()
        best_case = model.fit()
        output = pd.DataFrame(columns=['best_iter', 'best_hr', 'best_ndcg'])
        output.loc[i] = best_case
    output.to_csv(output_dir)


if __name__ == '__main__':
    NUM_ITER = 10
    result_out_file1 = 'outputs/best_GMF-8.csv'
    result_out_file2 = 'outputs/best_MGMF-8.csv'
    run_model(GMF, result_out_file1, num_iter=10)
    run_model(MGMF, result_out_file2, num_iter=10)
