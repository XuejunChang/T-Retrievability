import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import fair_utils
import os
import sys
import config

models = sys.argv[1:]
if __name__ == '__main__':
    models_coll_gini = {}
    for modelname in models:
        print(f'calc {modelname}')
        rscore_csv = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'
        if os.path.exists(rscore_csv):
            print(f'loading {rscore_csv}')
            df = pd.read_csv(rscore_csv, index_col=0).reset_index()
        else:
            result_csv = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}.csv'
            df = pd.read_csv(result_csv, index_col=0).reset_index()
            print('calc r_score of each document of the retrieved docs ...')
            df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))
            print(f'saving into {rscore_csv}')
            df.to_csv(rscore_csv, index=False)
            print(f'done')

        summed_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()
        print('calc coll_gini')
        coll_gini = fair_utils.compute_gini(summed_doc_rscores['r_score'].to_dict())
        print(f'{modelname}: {coll_gini}')
        models_coll_gini[modelname] = coll_gini
        results_coll_df = pd.DataFrame([models_coll_gini])

        result_csv_path = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_models_coll_gini.csv'
        if os.path.exists(result_csv_path):
            os.remove(result_csv_path)
            print(f'{result_csv_path} removed')
        print(f'saving into {result_csv_path}')
        results_coll_df.to_csv(result_csv_path, index=False)
        print('done')
