import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, pandas as pd

prog_dir = '/mnt/primary/exposure-fairness'
data_dir = '/mnt/datasets/cxj/exposure-fairness'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dataset_name = 'msmarco-passage'
topics_name = 'dev'

dataset = pt.get_dataset(f'irds:{dataset_name}')
dev = pt.get_dataset(f'irds:msmarco-passage/dev')
# corpus_df  = pd.DataFrame(dev.get_corpus_iter(verbose=True))
topics = dev.get_topics()
qrels = dev.get_qrels()

# dl19 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019')
# dl19_topics = dl19.get_topics()
# dl20 = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')
# dl20_topics = dl20.get_topics()
# dl1920_topics = pd.concat([dl19_topics, dl20_topics], ignore_index=True)
retrieve_num = 100
num_clusters = [500, 1000, 2000, 5000, 10000]
models = ['bm25', 'splade', 'tctcolbert', 'bm25_tctcolbert', 'bm25_monot5']
kmeans_vec = ['scikit_dense','scikit_tfidf']
# bm25 splade tctcolbert bm25_tctcolbert bm25_monot5