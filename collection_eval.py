import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import fair_utils

import os
import sys
from pathlib import Path
import gini
import config


def get_every_doc_rscore(csv_path):
    rscore_csv = os.path.splitext(csv_path)[0] + '_rscore.csv'
    if os.path.exists(rscore_csv):
        print(f'loading {rscore_csv}')
        df = pd.read_csv(rscore_csv, index_col=0).reset_index()
        print('done')
    else:
        df = pd.read_csv(csv_path, index_col=0).reset_index()
        print('calc r_score of each document of the retrieved docs ...')
        df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))
    
        print(f'saving into {rscore_csv}')
        df.to_csv(rscore_csv, index=False)
        print(f'done')

    return df
    
models_coll_gini = {}
def compute_each_model_coll_gini(modelname, csv_path):
    df = get_every_doc_rscore(csv_path)
    summed_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()    
    print('calc coll_gini')
    coll_gini = gini.compute_gini(summed_doc_rscores['r_score'].to_dict())
    print('done')
    
    print(f'{modelname}: {coll_gini}')
    # models_coll_gini[modelname] = coll_gini
    # results_coll_df = pd.DataFrame([models_coll_gini])

    # csv_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_models_coll_gini.csv'
    # if os.path.exists(csv_path):
    #     os.remove(csv_path)
    #     print(f'{csv_path} removed')
    # print(f'saving into {csv_path}')
    # results_coll_df.to_csv(csv_path, index=False)
    # print('done')


# args: [version] [run_model ... ]
# version = sys.argv[1]
# data_dir = f'{config.data_dir}/{version}'
# os.makedirs(data_dir, exist_ok=True)

# run_model = sys.argv[2:]
# topic = 'dev'

if __name__ == "__main__":
    # for modelname in run_model:
    #     print(f'calc {modelname}')
    #     rscore_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'
    #     compute_each_model_coll_gini(modelname, rscore_csv)

    # modelname = "cross-encoder"
    # rscore_csv  = "/mnt/datasets/cxj/fair-ranking-model/v1/reranked_df_no_trained.csv"
    # compute_each_model_coll_gini(modelname, rscore_csv)

    modelname = "raw-cross-encoder"
    csv_path  = "/mnt/datasets/cxj/fair-ranking-model/v1/reranked_df_trained.csv"
    collection_eval.compute_each_model_coll_gini(modelname, csv_path)

    

    
        