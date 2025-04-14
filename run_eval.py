import sys
import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from Lz4PickleCache import *
import pandas as pd
import os
import numpy as np

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
dev_topics = eval_dataset.get_topics()
dev_qrels = eval_dataset.get_qrels()

dl19 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019')
dl19_topics = dl19.get_topics()
dl20 = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')
dl20_topics = dl20.get_topics()
dl1920_topics = pd.concat([dl19_topics, dl20_topics], ignore_index=True)

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent


nfs_dir = f'/nfs/resources/cxj/retrievability-bias'
modelname = 't5-base-msmarco-epoch-5'
cache_file = Lz4PickleCache(f'/nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4')

pert_file = f'{nfs_dir}/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30/5, 60/5,90/5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

import bm25
import bm25_monot5
import rtr_splade
import tctcolbert
import bm25_tctcolbert

retrieve_num = 100
# rankername = sys.argv[1]
# os.makedirs(f'{nfs_dir}/{rankername}',exist_ok=True)
# topics = sys.argv[2]
# for inx, threshold in enumerate(percent):
#         eval(rankername).evaluate_experiment(inx, threshold, modelname, dataset, rankername, topics,retrieve_num, nfs_dir)

# rankername_grp = ['bm25', 'tctcolbert']
# rankername_grp = ['tctcolbert']
# rankername_grp = ['rtr_splade']
rankername_grp = ['bm25_monot5', 'bm25_tctcolbert']

topics_grp = ['dev_topics','dl1920_topics']
# topics_grp = ['dev_topics']
# topics_grp = ['dl1920_topics']

for rankername in rankername_grp:
    ranker =  eval(rankername)
    for topics in topics_grp:
        for inx, threshold in enumerate(percent):
            # if rankername == 'tctcolbert' and topics == 'dev_topics' and inx == 0:
            #     continue
            print(f'start {rankername} --> {topics}')
            ranker.evaluate_experiment(inx, threshold, modelname, dataset, rankername, topics,eval(topics),retrieve_num, nfs_dir)



