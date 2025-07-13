import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, pandas as pd

index_dir = '/mnt/datasets/cxj/exposure-fairness/v2'
prog_dir = '/mnt/primary/exposure-fairness-extend'
data_dir = '/mnt/datasets/cxj/exposure-fairness-extend'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# prog_dir = '/nfs/primary/exposure-fairness'
# data_dir = '/nfs/datasets/cxj/exposure-fairness'

models = ['bm25', 'splade', 'tctcolbert', 'bm25_tctcolbert', 'bm25_monot5']
# bm25 splade tctcolbert bm25_tctcolbert bm25_monot5

dataset_name = 'msmarco-passage'
topics_name = 'dev'
dataset = pt.get_dataset(f'irds:{dataset_name}')
dev = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = dev.get_topics()
qrels = dev.get_qrels()

corpus_df  = pd.DataFrame(dev.get_corpus_iter(verbose=True))

# dl19 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019')
# dl19_topics = dl19.get_topics()
# dl20 = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')
# dl20_topics = dl20.get_topics()
# dl1920_topics = pd.concat([dl19_topics, dl20_topics], ignore_index=True)

num_clusters = [500, 1000, 2000, 5000, 10000]