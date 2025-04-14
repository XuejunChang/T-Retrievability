import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

# from DocumentFilter import *
from Lz4PickleCache import *
import pandas as pd
import os
import shutil
import pyterrier_dr
# rankername = os.path.basename(__file__).split('.')[0]

import bm25
import tctcolbert
def evaluate_experiment(inx, threshold, modelname, dataset, rankername, topics, topics_ins, retrieve_num, nfs_dir):
    index_path = f"{nfs_dir}/tctcolbert/{modelname}-tctcolbert-index-threshold-{threshold}.flex"
    if not os.path.exists(index_path):
        print(f"Index file {index_path} does not exist")
        return

    csv = f'{nfs_dir}/{rankername}/df_{rankername}_{topics}_{inx * 30}.csv'
    if os.path.exists(csv):
        print(f'file {csv} already exists')
        # df = pd.read_csv(csv, index_col=0).reset_index()
        return
    else:
        bm25_csv = bm25.evaluate_experiment(inx, threshold, modelname, dataset, "bm25", topics, topics_ins, retrieve_num, nfs_dir)
        bm25_df = pd.read_csv(bm25_csv, index_col=0).reset_index()
        bm25_df[['qid', 'docno']] = bm25_df[['qid', 'docno']].astype(str)

        print(f'start tramsforming {rankername} {topics} at {inx * 30}%')

        retr_pipeline = pt.text.get_text(dataset, 'text', verbose=True) >> tctcolbert.model
        df = retr_pipeline.transform(bm25_df)

        print(f'df of {rankername} {topics} columns {df.columns.tolist()}')
        cols = ['qid', 'docid', 'docno', 'score', 'rank']
        print(f'opt in columns {cols}')
        df = df[cols]
        df.to_csv(csv, index=False)
        print(f'saved {rankername} {topics} with shape {df.shape} into {csv}')







