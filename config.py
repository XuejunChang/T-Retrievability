import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, pandas as pd

def save_trec_qrels(df, save_file):
    if not os.path.exists(save_file):
        print(f'saving into {save_file}')
        result = pd.DataFrame()
        result['query_id'] = df['qid']
        result['Q0'] = 0
        result['doc_id'] = df['docno']
        result['relevance'] = df['label']

        result.to_csv(save_file, sep=' ', index=False, header=False)
        print(f'saved')

# prog_dir = '/mnt/primary/exposure-fairness'
# data_dir = '/mnt/datasets/cxj/exposure-fairness'
prog_dir = '/nfs/primary/exposure-fairness'
data_dir = '/nfs/datasets/cxj/exposure-fairness'

# bm25 bm25_monot5 splade tctcolbert bm25_tctcolbert

dataset_name = 'msmarco-passage'
topics_name = 'dev'
retrieve_num = 100
dataset = pt.get_dataset(f'irds:{dataset_name}')
dev = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = dev.get_topics()

qrels = dev.get_qrels()
trec_qrels = f'{data_dir}/qrels_dev.res'
save_trec_qrels(qrels, trec_qrels)

# dl19 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019')
# dl19_topics = dl19.get_topics()
# dl20 = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')
# dl20_topics = dl20.get_topics()
# dl1920_topics = pd.concat([dl19_topics, dl20_topics], ignore_index=True)