import GMF
import MGMF
import NeuMF
import MNeuMF
import MLP
import pandas as pd
from data_processing.ml_utils import load_100k, load_1m


def run_model_case(model, output, data_name, batch_size):
    best_case = model.fit(data_name, batch_size)
    #output.loc[i] = best_case
    return best_case
    


if __name__ == '__main__':
    data_name = '1m'
    batch_size = 256
    NUM_ITER = 1
    run_model = lambda model, output: run_model_case(model, output, data_name, batch_size)

    #result_out_file1 = 'outputs/best_GMF-8.csv'
    result_out_file2 = 'outputs/best_MGMF-8.csv'
    result_out_file3 = 'outputs/best_NeuMF-8.csv'
    #result_out_file4 = 'outputs/best_MNeuMF-8.csv'
    #result_out_file5 = 'outputs/best_MLP-8.csv'
    #output_gmf = pd.DataFrame(columns=['best_iter', 'best_hr', 'best_ndcg'])
    output_mgmf = pd.DataFrame(columns=['best_iter', 'best_hr', 'best_ndcg'])
    output_neumf = pd.DataFrame(columns=['best_iter', 'best_hr', 'best_ndcg'])
    #output_mneumf = pd.DataFrame(columns=['best_iter', 'best_hr', 'best_ndcg'])
    #output_mlp = pd.DataFrame(columns=['best_iter', 'best_hr', 'best_ndcg'])
    for i in range(NUM_ITER):
        if data_name == '100k':
            load_100k()
        else:
            load_1m()

        #output_gmf.loc[i] = run_model(GMF, output_gmf)
        output_neumf.loc[i] = run_model(NeuMF, output_neumf)
        output_mgmf.loc[i] = run_model(MGMF, output_mgmf)
        
        #output_mneumf.loc[i] = run_model(MNeuMF, output_mneumf)
        #output_mlp.loc[i] = run_model(MLP, output_mlp)
        print("+"*40 + " Iteration {} finished ".format(i) + "+"*40)
    
    #output_gmf.to_csv(result_out_file1)
    output_mgmf.to_csv(result_out_file2)
    output_neumf.to_csv(result_out_file3)
    #output_mneumf.to_csv(result_out_file4)
    #output_mlp.to_csv(result_out_file5)
    
