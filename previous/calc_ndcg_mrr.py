import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

import os
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', False)

import ir_measures
from ir_measures import * # imports all supported measures, e.g., AP, nDCG, RR, P


import ir_datasets
qrels = ir_datasets.load("msmarco-passage/dev").qrels_iter()

bm25 = pd.read_csv('/nfs/datasets/cxj/retrievability-bias/results_bm25_10.csv', index_col=0)

bm25_run = bm25.rename(columns={'qid':'query_id','docid':'doc_id'})
del bm25_run['score']
bm25_run['score'] = 10-bm25_run['rank']
bm25_run[['query_id','doc_id']] = bm25_run[['query_id','doc_id']].astype(str)
metrics = ir_measures.calc_aggregate([nDCG@10, RR], qrels, bm25_run[:10])
print(metrics)