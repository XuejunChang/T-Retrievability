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

def create_index(inx, threshold, modelname, cache_file, nfs_dir, qual_signal=None):
    index_file = f"{nfs_dir}/bm25/{modelname}-nostemmer-nostopwords-index-{str(threshold)}"

    print(f"indexing into {index_file}")
    indexer = (DocumentFilter(qual_signal=qual_signal, threshold=threshold)
               >> pt.IterDictIndexer(index_file,stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, verbose=True))
    indexref = indexer.index(pt.tqdm(cache_file.get_corpus_iter()))

    print('indexing done')

def evaluate_experiment(inx, threshold, modelname, dataset, rankername, topics, topics_ins, retrieve_num, nfs_dir):
    index_file = f"{nfs_dir}/{rankername}/{modelname}-nostemmer-nostopwords-index-{str(threshold)}"
    # cache_dir = f"{nfs_dir}/{rankername}/{rankername}_cache_{str(threshold)}"
    # cached_bm25 = utils.get_cached_bm25(index_file, cache_dir)
    retriever = pt.terrier.Retriever(index_file, wmodel='BM25', verbose=True)
    retriever = retriever % retrieve_num

    bm25_csv = f'{nfs_dir}/{rankername}/df_{rankername}_{topics}_{inx * 30}.csv'
    if os.path.exists(bm25_csv):
       return bm25_csv
    else:
        print(f'tramsforming {rankername} {topics} at {inx * 30}%')
        df = retriever.transform(topics_ins)
        print(f'df of {rankername} columns {df.columns.tolist()}')
        cols = ['qid','docid','docno','score','rank','query']
        print(f'to to opt in in columns {cols}')
        df = df[cols]
        df.to_csv(bm25_csv,index=False)
        print(f'saved {rankername} {topics} with shape {df.shape} into {bm25_csv}')

