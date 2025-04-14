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

ranker = "tctcolbert"
retrieve_num = 100

import pyterrier_dr
import shutil

model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
def create_index(threshold, modelname, iterable_dict):
    index_path = f"{nfs_dir}/{ranker}/{modelname}-{ranker}-index-threshold-{threshold}.flex"
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        print(f'existing index file at {index_path} removed')

    print(f'indexing into {index_path}')
    index = pyterrier_dr.FlexIndex(f'{index_path}')

    idx_pipeline =  model >> index
    idx_pipeline.index(iterable_dict)
    print(f'indexing {index_path} done')

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

modelname = 't5-base-msmarco-epoch-5'
root_file_name = f'{root_dir}/{modelname}.lz4'

# if not os.path.exists(root_file_name):
#     os.system(f'cp /nfs/resources/cxj/retrievability-bias/supervisedT5/npy121pruned/{modelname}.lz4 {root_dir}/')
    # os.system(f'cp /nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4 {root_dir}/')
    # os.system(f'cp /nfs/resources/cxj/distill/distill/trained_models/{modelname}.lz4 {root_dir}/')
cache_file = Lz4PickleCache(f'{root_dir}/{modelname}.lz4')

pert_file = f'{nfs_dir}/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30/5, 60/5, 90/5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)
    print('saved percentage')

import sys
p = int(sys.argv[1])
for i, threshold in enumerate(percent):
    if i == p:
        csv = f'{nfs_dir}/{modelname}_prunned_{30 * i}_df.csv'
        df = pd.read_csv(csv, index_col=0).reset_index()
        del df['prob']
        df[['docno', 'text']] = df[['docno', 'text']].astype(str)

        # rng = [str(docno) for docno in range(8840985, 8841820)]
        # rng = [str(docno) for docno in range(8840800, 8840985)] # good
        # iterable_dict = df[df['docno'].isin(rng)].to_dict(orient="records")
        iterable_dict = df.to_dict(orient="records")
        create_index(threshold, modelname, iterable_dict)






