import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from Lz4PickleCache import *

import pandas as pd
from pyterrier_caching import ScorerCache
from pyterrier_t5 import MonoT5ReRanker
import os
import numpy as np

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

ranker = "bm25"
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

monot5 = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(batch_size=16)
cached_monot5 = ScorerCache(f'{nfs_dir}/{ranker}/msmarco-passage.monot5-base.cache', monot5)

def evaluate_experiment(inx, threshold, modelname):
    index_file = f"{nfs_dir}/{ranker}/{modelname}-nostemmer-nostopwords-index-{str(threshold)}"
    # cache_dir = f"{nfs_dir}/{ranker}/{ranker}_cache_{str(threshold)}"
    # cached_bm25 = utils.get_cached_bm25(index_file, cache_dir)
    retriever = pt.terrier.Retriever(index_file, wmodel='BM25', verbose=True)
    retriever = retriever % retrieve_num

    print('start tramsforming {ranker}')
    csv = f'{nfs_dir}/{ranker}/df_{ranker}_{inx * 30}.csv'
    if os.path.exists(csv):
        df = pd.read_csv(csv,index_col=1).reset_index()
    else:
        df = retriever.transform(topics)
        print(f'df of {ranker} columns {df.columns.tolist()}')
        cols = ['qid','docid','docno','score','rank','query']
        print(f'to to opt in in columns {cols}')
        df = df[cols]
        df.to_csv(csv,index=False)
        print(f'save {ranker} with shape {df.shape} into {csv}')

    # print(f'start calc {ranker} metrics')
    # mean, std_dev, gini_value = utils.calc_stats(ranker, df, threshold, topics)
    # nDCG, RR = utils.calc_irmetrics(df)
    # print(f'{ranker} mean: {mean}, std_dev: {std_dev}, gini_value:{gini_value}')
    # print(f'{ranker} nDCG: {nDCG}, RR: {RR}')
    #
    # result_df = pd.DataFrame([[mean, std_dev, gini_value, nDCG, RR]],columns=['mean', 'std_dev', 'gini_value', 'nDCG', 'RR'])
    # csv = f'{nfs_dir}/{ranker}/result_{ranker}_{inx*30}.csv'
    # result_df.to_csv(csv,index=False)
    # print(f'save to {csv}')

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

modelname = 't5-base-msmarco-epoch-5'
cache_file = Lz4PickleCache(f'/nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4')

pert_file = f'{nfs_dir}/{ranker}/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30 / 5, 60 / 5, 90 / 5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

for i, threshold in enumerate(percent):
    if i == 0:
        continue
    evaluate_experiment(i, threshold, modelname)
