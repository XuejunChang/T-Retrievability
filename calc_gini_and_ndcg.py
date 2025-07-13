import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, sys
import numpy as np, pandas as pd
from tqdm import tqdm
import torch
import config
import collection_eval
import gini
import fair_utils

# csv_file  = f'{config.data_dir}/v1/bm25_msmarco-passage_dev_100.csv'
# csv_file  = f'{config.data_dir}/v1/splade_msmarco-passage_dev_100_diver_df.csv'
# csv_file  = f'{config.data_dir}/v1/tctcolbert_msmarco-passage_dev_100_diver_df.csv'
# csv_file  = f'{config.data_dir}/v1/bm25_tctcolbert_msmarco-passage_dev_100_diver_df.csv'
# csv_file  = f'{config.data_dir}/v1/bm25_monot5_msmarco-passage_dev_100_diver_df.csv'

modelname = sys.argv[1]
res_file = sys.argv[2] # tctcolbert_msmarco-passage_dev_200_diver_df_cut100.res
res_path = f'{config.data_dir}/{res_file}'

csv_file = f'{os.path.splitext(res_path)[0]}.csv'
df = collection_eval.get_every_doc_rscore(csv_file)
summed_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()    
print('calc coll_gini')
coll_gini = gini.compute_gini(summed_doc_rscores['r_score'].to_dict())
print(f'{modelname} coll_gini: {coll_gini:.4f}')

qrels_res = f'{config.data_dir}/qrels_dev.res'
result = fair_utils.cal_metrics(qrels_res, res_path)
print(f"{modelname} nDCG@10: {result['ndcg_cut_10']:.4f}, map: {result['map']:.4f}")