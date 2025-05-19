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

def calc_metrics(sum_doc_rscores):
    print('calc coll_gini')
    coll_gini = gini.compute_gini(sum_doc_rscores['r_score'].to_dict())
    print('done')

    return coll_gini

# args: [version] [delcache] [run_model ... ]
version = sys.argv[1]
data_dir = f'{config.data_dir}/{version}'
os.makedirs(data_dir, exist_ok=True)

delcache = True if sys.argv[2] == 'True' else False
run_model = sys.argv[3:]

num_clusters = [500, 1000, 2000, 5000, 10000]
topic = 'dev'
models_coll_gini = {}
for modelname in run_model:
    print(f'calc {modelname}')
    rscore_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'
    if delcache:  # all caches refer to files related to r_score.
        Path(rscore_csv).unlink(missing_ok=True)
        print(f'{rscore_csv} removed')
    if os.path.exists(rscore_csv):
        print(f'loading {rscore_csv}')
        df = pd.read_csv(rscore_csv, index_col=0).reset_index()
    else:
        result_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}.csv'
        df = pd.read_csv(result_csv, index_col=0).reset_index()
        print('calc r_score of each document of the retrieved docs ...')
        df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))
        print(f'saving into {rscore_csv}')
        df.to_csv(rscore_csv, index=False)
        print(f'done')

    summed_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()
    coll_gini = calc_metrics(summed_doc_rscores)
    print(f'{modelname}: {coll_gini}')
    models_coll_gini[modelname] = coll_gini
    results_coll_df = pd.DataFrame([models_coll_gini])

    result_csv_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_models_coll_gini.csv'
    if os.path.exists(result_csv_path):
        os.remove(result_csv_path)
        print(f'{result_csv_path} removed')
    print(f'saving into {result_csv_path}')
    results_coll_df.to_csv(result_csv_path, index=False)
    print('done')




