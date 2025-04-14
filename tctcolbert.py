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
model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco', verbose=True)
def create_index(threshold, modelname, rankername, iterable_dict, nfs_dir):
    index_path = f"{nfs_dir}/{rankername}/{modelname}-{rankername}-index-threshold-{threshold}.flex"
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        print(f'existing index file at {index_path} removed')

    print(f'indexing into {index_path}')
    index = pyterrier_dr.FlexIndex(f'{index_path}')
    idx_pipeline =  model >> index
    idx_pipeline.index(iterable_dict)
    print(f'indexing {index_path} done')

def evaluate_experiment(inx, threshold, modelname, dataset, rankername, topics, topics_ins, retrieve_num, nfs_dir):
    index_path = f"{nfs_dir}/{rankername}/{modelname}-{rankername}-index-threshold-{threshold}.flex"
    if not os.path.exists(index_path):
        print(f"Index file {index_path} does not exist")
        return

    csv = f'{nfs_dir}/{rankername}/df_{rankername}_{topics}_{inx * 30}.csv'
    if os.path.exists(csv):
        print(f'csv file {csv} already exists')
        # df = pd.read_csv(csv, index_col=0).reset_index()
        return
    else:
        print(f'start tramsforming {rankername} {topics} at {inx * 30}%')
        index = pyterrier_dr.FlexIndex(index_path)
        retr_pipeline = model >> index.torch_retriever() % retrieve_num
        df = retr_pipeline.transform(topics_ins)
        print(f'df of {rankername} {topics} columns {df.columns.tolist()}')
        cols = ['qid', 'docid', 'docno', 'score', 'rank']
        print(f'opt in columns {cols}')
        df = df[cols]
        df.to_csv(csv, index=False)
        print(f'saved {rankername} {topics} with shape {df.shape} into {csv}')








