import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from DocumentFilter import *
from Lz4PickleCache37 import *
import pandas as pd
import os
import numpy as np

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

ranker = "colbert"
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()


from pyterrier_colbert.indexing import ColBERTIndexer
CHECKPOINT="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
def create_index(threshold, modelname, cache_file, qual_signal=None):
    index_path = f"{nfs_dir}/{ranker}/{modelname}-{ranker}-index-threshold-{threshold}"
    print(f'indexing into {index_path}')
    indexer = ColBERTIndexer(CHECKPOINT, f"{nfs_dir}/colbert/", f"{modelname}-{ranker}-index-threshold-{threshold}", chunksize=8, gpu=True)
    pipe = DocumentFilter(qual_signal=qual_signal, threshold=threshold) >> indexer
    pipe.index(pt.tqdm(cache_file.get_corpus_iter()))

    print('indexing done')

from pyterrier_colbert.ranking import ColBERTFactory
def evaluate_experiment(inx, threshold, modelname,qstart,qend):
    index_path = f"{nfs_dir}/{ranker}/{modelname}-{ranker}-index-threshold-{threshold}"
    if not os.path.exists(index_path):
        print(f"Index file {index_path} does not exist")
        return

    index_dir = '/'.join(index_path.split('/')[:-1])
    index_name = index_path.split('/')[-1]
    pytcolbert = ColBERTFactory(CHECKPOINT, f"{index_dir}", index_name, faiss_partitions=100)
    retriever = pytcolbert.end_to_end() % retrieve_num

    csv = f'{nfs_dir}/{ranker}/df_{ranker}_{inx * 30}_{qstart}.csv'
    if os.path.exists(csv):
        print(f'{ranker} csv file {csv} already exists')
        df = pd.read_csv(csv, index_col=0).reset_index()
        return df
    else:
        print(f'start tramsforming {ranker} with threshold {threshold}')
        if qend == -1:
            qend = topics.shape[0]
        df = retriever.transform(topics[qstart:qend])
        print(f'df of {ranker} columns {df.columns.tolist()}')
        cols = ['qid', 'docid', 'docno', 'score', 'rank']
        print(f'to opt in columns {cols}')
        df = df[cols]
        df.to_csv(csv, index=False)
        print(f'saved {ranker} with shape {df.shape} into {csv}')

    return df

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

result_df_colbert = pd.DataFrame()
modelname = 't5-base-msmarco-epoch-5'

root_file_name = f'{root_dir}/{modelname}.lz4'
if not os.path.exists(root_file_name):
    os.system(f'cp /nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4 {root_dir}/')
    # os.system(f'cp /nfs/resources/cxj/distill/distill/trained_models/{modelname}.lz4 {root_dir}/')
cache_file = Lz4PickleCache(f'{root_dir}/{modelname}.lz4')

pert_file = f'{nfs_dir}/bm25/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30/5, 60/5, 90/5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

import sys
qstart,qend = int(sys.argv[1]), int(sys.argv[2])
for i, threshold in enumerate(percent):
    if i != 0:
        continue
    create_index(threshold, modelname, cache_file, qual_signal="prob")
    evaluate_experiment(i, threshold, modelname, qstart, qend)





