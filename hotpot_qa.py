import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import numpy as np, pandas as pd
import os, sys, time

def create_index(modelname, data_dir, dataset_name, dataset):
    index_path = f"{data_dir}/indices/{dataset_name}-{modelname}-nostemmer-nostopwords-index"
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    else:
        return index_path

    print(f"indexing into {index_path}")
    indexer = pt.IterDictIndexer(index_path,stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, verbose=True)
    start = time.time()
    indexref = indexer.index(dataset.get_corpus_iter(verbose=True))
    end = time.time()
    print(f'indexing done in {end-start}/60 minutes')
    return indexref

data_dir = '/nfs/datasets/cxj/retrievability-bias/data'
modelname = 'BM25'
dataset_name = 'hotpotqa'
dataset = pt.get_dataset(f'irds:beir/{dataset_name}') # 5233329 docs
index_ref = create_index(modelname, data_dir, dataset_name, dataset)

print('start hotpotqa retrieval ')
pipeline = pt.terrier.Retriever(index_ref, wmodel='BM25') % 100 >> pt.text.get_text(dataset, 'text')
topics = dataset.get_topics()
print(pipeline.transform(topics[:10]))

print('start msmarco retrieval ')
dataset_name = 'msmarco-passage'
dataset = pt.get_dataset(f'irds:{dataset_name}')
index_ref = create_index(modelname, data_dir, dataset_name, dataset)
dev = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = dev.get_topics()
pipeline = pt.terrier.Retriever(index_ref, wmodel='BM25') % 100 >> pt.text.get_text(dataset, 'text')
print(pipeline.transform(topics[:10]))


