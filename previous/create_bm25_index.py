

import pyterrier as pt
if not pt.started():
    pt.init()

from DocumentFilter import *
from Lz4PickleCache import *

import pandas as pd

from pyterrier_t5 import MonoT5ReRanker

import os
import numpy as np

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/datasets/cxj/retrievability-bias'

ranker = "BM25" 
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()


monot5 = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(batch_size=16)
cached_monot5 = ScorerCache(f'{nfs_dir}/bm25/msmarco-passage.monot5-base.cache', monot5)

def create_index(inx, threshold, modelname, index_path, expt_path, cache_file, qual_signal=None):
    index_file = f"{nfs_dir}/bm25/{modelname}-nostemmer-nostopwords-index-{str(threshold)}"

    print(f"indexing into {index_file}")
    indexer = (DocumentFilter(qual_signal=qual_signal, threshold=threshold)
               >> pt.IterDictIndexer(index_file,stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, verbose=True))
    indexref = indexer.index(pt.tqdm(cache_file.get_corpus_iter()))

    print('indexing done')


def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

result_df_BM25 = pd.DataFrame()
result_df_BM25_MonoT5 = pd.DataFrame()

modelname = 't5-base-msmarco-epoch-5'
index_path = f'{root_dir}/msmarco_dev-{ranker}'
expt_path = f'{root_dir}/msmarco_dev-{ranker}'

root_file_name = f'{root_dir}/{modelname}.lz4'
if not os.path.exists(root_file_name):
    os.system(f'cp /nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4 {root_dir}/')
cache_file = Lz4PickleCache(f'{root_dir}/{modelname}.lz4')

pert_file = f'{nfs_dir}/bm25/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30 / 5, 60 / 5, 90 / 5])  # take 30%, 60%, 90% pruned
    # percent = np.array(percent[:-1]).take([0, 30 / 5, 60 / 5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

for i, threshold in enumerate(percent):
    if i == 0:
        continue
    evaluate_experiment(i, threshold, modelname, index_path, expt_path, cache_file, qual_signal="prob")


