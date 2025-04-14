import sys
import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()
import faiss
import datetime
import os
import numpy as np
import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', False)
from contriever_model import Contriever
import bm25_retrieve as bm25retr

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/datasets/cxj/retrievability-bias'
if not os.path.exists(nfs_dir):
    os.makedirs(nfs_dir)

model_name = "facebook/contriever-msmarco"
model = Contriever(model_name, batch_size=64)

if __name__ == '__main__':
    print(f'{datetime.datetime.now()}:start to retrieve')
    nopruned_index = f"{nfs_dir}/msmarco-passage-nostemmer-nostopwords-index"
    bm25_cache_dir = f"/nfs/datasets/cxj/retrievability-bias/bm25/bm25_cache"
    br1 = bm25retr.get_cached_bm25(nopruned_index, bm25_cache_dir)

    retr_pipeline = br1 % 100 >> pt.text.get_text(dataset,'text') >> model
    results = retr_pipeline.transform(topics[0:2])
    print(results.columns)

    # csv = f'results_bm25_contriever_100.csv'
    # results.to_csv(f'{root_dir}/{csv}')
    # os.system(f'cp -r {root_dir}/{csv} {nfs_dir}/')
    # print(f'copied {root_dir}/{csv} into {nfs_dir}')


