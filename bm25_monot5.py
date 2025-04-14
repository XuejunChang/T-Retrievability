import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from Lz4PickleCache import *
import DocumentFilter
import pandas as pd
from pyterrier_t5 import MonoT5ReRanker
import os
import bm25

def evaluate_experiment(inx, threshold, modelname, dataset, rankername, topics, topics_ins, retrieve_num, nfs_dir):
    monot5_csv = f'{nfs_dir}/{rankername}/df_{rankername}_{topics}_{inx * 30}.csv'
    if not os.path.exists(monot5_csv):
        bm25_csv = bm25.evaluate_experiment(inx, threshold, modelname, dataset, "bm25", topics, topics_ins, retrieve_num, nfs_dir)
        bm25_df = pd.read_csv(bm25_csv, index_col=0).reset_index()
        bm25_df[['qid', 'docno']] = bm25_df[['qid', 'docno']].astype(str)
        monot5 = pt.text.get_text(dataset, 'text', verbose=True) >> MonoT5ReRanker(batch_size=16)

        print(f'start tramsforming {rankername} {topics} at {inx * 30}%')
        df = monot5.transform(bm25_df)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid', 'docid', 'docno', 'score', 'rank']]
        print(f'to to opt in in shape {df.shape}')

        df.to_csv(monot5_csv, index=False)
        print(f'saved into {monot5_csv}')
