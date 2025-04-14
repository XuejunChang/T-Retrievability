import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

import os
import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', False)

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

root_dir = f'/root/retrievability-bias'
nfs_save = f'/nfs/datasets/cxj/retrievability-bias'
if not os.path.exists(nfs_save):
    os.makedirs(nfs_save)

# def create_indexes():
#     index_file = f"{root_dir}/msmarco-passage-nostemmer-nostopwords-index"
#     nfs_index_file = f"{nfs_save}/msmarco-passage-nostemmer-nostopwords-index"
#     if os.path.exists(index_file):
#         index_ref = pt.IndexRef.of(index_file)
#     else:
#         if os.path.exists(nfs_index_file):
#             os.system(f'cp -r {nfs_index_file}/ {root_dir}/')
#             print(f'copied from {nfs_save}')
#             index_ref = pt.IndexRef.of(index_file)
#         else:
#             print(f"indexing into {index_file}")
#             indexer = pt.IterDictIndexer(index_file, stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, verbose=True)
#             index_ref = indexer.index(dataset.get_corpus_iter(verbose=True))
#             os.system(f'cp -r {index_file} {nfs_save}/')
#             print(f'copied index into {nfs_save}')
#
#     index = pt.IndexFactory.of(index_ref)
#     return index

def create_indexes(index_file):
    if not os.path.exists(index_file):
        print(f"indexing into {index_file}")
        indexer = pt.IterDictIndexer(index_file, stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, verbose=True)
        indexer.index(dataset.get_corpus_iter(verbose=True))
    return index_file

import numpy as np

if __name__ == '__main__':
    index_name = "msmarco-passage-nostemmer-nostopwords-index"
    bm25_cache_dir = f"/nfs/datasets/cxj/retrievability-bias/bm25/bm25_cache"
    cached_bm25 = get_cached_bm25(index_name, bm25_cache_dir)
    # br = pt.terrier.Retriever(index, wmodel='BM25',verbose=True) % 100
    # pipe = cached_bm25 % 100 >> pt.text.get_text(dataset,'text', verbose=True)
    pipe = cached_bm25 % 100
    print('start bm25 retrieval')
    result = pipe.transform(pt.tqdm(topics))
    # print(result)

    csvfile = 'results_bm25_100.csv'
    result.to_csv(csvfile,index=False)
    os.system(f'cp -r {root_dir}/{csvfile} {nfs_save}/contrast/')
    print(f'copied {root_dir}/{csvfile} into {nfs_save}/contrast/')

