import sys
import warnings

warnings.filterwarnings('ignore')

import pyterrier as pt
if not pt.started():
    pt.init()

from DocumentFilter import *
from Lz4PickleCache import *

import pandas as pd

import os
import numpy as np

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'



modelname = 't5-base-msmarco-epoch-5'
root_file_name = f'{root_dir}/{modelname}.lz4'
if not os.path.exists(root_file_name):
    os.system(f'cp /nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4 {root_dir}/')
cache_file = Lz4PickleCache(f'{root_dir}/{modelname}.lz4')


def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

pert_file = f'{nfs_dir}/bm25/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30/5, 60/5, 90/5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

# p = int(sys.argv[1])
for i, threshold in enumerate(percent):
    if i==3:
        csv = f'{nfs_dir}/{modelname}_prunned_{30*i}_df.csv'
        if os.path.exists(csv):
            os.remove(csv)
            print(f'old {csv} removed')

        print('decompressing')
        doc_fiter = DocumentFilter(qual_signal='prob', threshold=threshold)
        res_df = doc_fiter(pt.tqdm(cache_file.get_corpus_iter()))

        print(f'writing {csv}')
        res_df.to_csv(csv, index=False)
        print('done')


