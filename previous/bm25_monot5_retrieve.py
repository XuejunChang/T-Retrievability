import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()

import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', False)

from pyterrier_caching import ScorerCache

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

work_name = "retrievability-bias"
root_dir = f'/root/{work_name}'
nfs_save = f'/nfs/datasets/cxj/{work_name}'
if not os.path.exists(nfs_save):
    os.makedirs(nfs_save)

import bm25_retrieve as bm25retr
from pyterrier_t5 import MonoT5ReRanker

# monot5 = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(batch_size=16)
# cached_monot5 = ScorerCache(f'{nfs_dir}/bm25/msmarco-passage.monot5-base.cache', monot5)

from pyterrier_caching import RetrieverCache
def get_cached_bm25(index_file, cache_dir):
    print(f'checking cache dir {cache_dir}')
    if not os.path.exists(cache_dir):
        print('start caching bm25 retrieval')
        retriever = pt.terrier.Retriever(index_file, wmodel='BM25', verbose=True)
        cached_bm25 = RetrieverCache(cache_dir, retriever)
    else:
        cached_bm25 = RetrieverCache(cache_dir)
    return cached_bm25

if __name__ == '__main__':
    bm25_index = "msmarco-passage-nostemmer-nostopwords-index"
    bm25_cache_dir = f"/nfs/datasets/cxj/retrievability-bias/bm25/bm25_cache"
    cached_bm25 = get_cached_bm25(bm25_index, bm25_cache_dir)

    nfs_monot5_cache_path = '/nfs/datasets/cxj/cache/msmarco-passage-monot5.cache'
    monot5 = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker()
    cached_monot5 = ScorerCache('testmonot5cache', monot5)

    mono_pipeline = cached_bm25 >> cached_monot5
    print('start bm25 >> monot5 retrieval')
    result = mono_pipeline.transform(topics[:2])
    print(result.head())

    # csv = f'results_bm25_monot5_100.csv'
    # result.to_csv(f'{root_dir}/{csv}')
    # os.system(f'cp -r {root_dir}/{csv} {nfs_save}/')
    # print(f'copied {root_dir}/{csv} into {nfs_save}')
