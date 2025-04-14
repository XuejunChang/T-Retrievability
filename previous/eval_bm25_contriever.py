import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from DocumentFilter import *
from Lz4PickleCache import *

import pandas as pd
from pyterrier_caching import ScorerCache
import utils
import os
import numpy as np
from pyterrier_caching import RetrieverCache
root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/datasets/cxj/retrievability-bias'

ranker = "contriever"
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

from contriever_model import Contriever
model_name = "facebook/contriever-msmarco"
model = Contriever(model_name, batch_size=16)

def evaluate_experiment(inx, threshold, modelname):
    index_file = f'{nfs_dir}/{ranker}/{modelname}-{ranker}-index-{str(threshold)}.flex'
    if not os.path.exists(index_file):
        print(f"Index file {index_file} does not exist")
        return

    cache_bm25_dir = f"{nfs_dir}/bm25/bm25_cache_{str(threshold)}"
    cached_bm25 = utils.get_cached_bm25(None, cache_bm25_dir)

    pipeline2 = cached_bm25  % retrieve_num >> pt.text.get_text(dataset,'text') >> model

    print('start tramsforming bm25 >> {ranker}')
    df = pipeline2.transform(topics)
    print('start calc bm25 >> {ranker} metrics')
    mean, std_dev, gini_value = utils.calc_stats(f'bm25_{ranker}', df, threshold, topics)
    nDCG, RR = utils.calc_irmetrics(df)
    print(f'bm25 >> {ranker} mean: {mean}, std_dev: {std_dev}, gini_value:{gini_value}')
    print(f'bm25 >> {ranker} nDCG: {nDCG}, RR: {RR}')

    result_df = pd.DataFrame([[mean, std_dev, gini_value, nDCG, RR]],columns=['mean', 'std_dev', 'gini_value', 'nDCG', 'RR'])
    csv = f'{nfs_dir}/{ranker}/result_bm25_{ranker}_{inx*30}.csv'
    result_df.to_csv(csv,index=False)
    print(f'save to {csv}')

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

modelname = 't5-base-msmarco-epoch-5'
cache_file = Lz4PickleCache(f'/nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4')

pert_file = f'{nfs_dir}/bm25/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30 / 5, 60 / 5, 90 / 5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

for i, threshold in enumerate(percent):
    # if i == 0:
    #     continue
    evaluate_experiment(i, threshold, modelname)
