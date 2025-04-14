import warnings

warnings.filterwarnings('ignore')

import pyterrier as pt
if not pt.started():
    pt.init()

from DocumentFilter import *
from Lz4PickleCache import *

import os
import numpy as np

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

ranker = "contriever"

from contriever_model import Contriever
import pyterrier_dr

model_name = "facebook/contriever-msmarco"
model = Contriever(model_name, batch_size = 32)


# def create_index(inx, threshold, modelname, qual_signal="prob"):
#     index_file = f'{nfs_dir}/{ranker}/{modelname}-{ranker}-index-{str(threshold)}_test.flex'
#     if os.path.exists(index_file):
#         shutil.rmtree(index_file)
#
#     print(f"indexing into {index_file}")
#     index = pyterrier_dr.FlexIndex(index_file)
#     csv = f'{nfs_dir}/bm25/{modelname}_unprunned_df.csv'
#     df = pd.read_csv(csv,index_col=0).reset_index()
#     pipe1 = DocumentFilter(qual_signal=qual_signal, threshold=threshold)
#     fitered_df = pipe1.transform(df)
#     pipe2 =  model >> index
#     pipe2.index(fitered_df)
#     print(f'index {threshold} done')

def create_index(inx, threshold, cache_file, modelname, qual_signal="prob"):
    index_file = f'{nfs_dir}/{ranker}/{modelname}-{ranker}-index-{str(threshold)}.flex'
    if os.path.exists(index_file):
        # shutil.rmtree(index_file)
        print(f"Index file {index_file} already exists, skipping")
        return

    print(f"indexing into {index_file}")
    index = pyterrier_dr.FlexIndex(index_file)
    idx_pipeline = DocumentFilter(qual_signal=qual_signal, threshold=threshold) >> model >> index
    idx_pipeline.index(pt.tqdm(cache_file.get_corpus_iter()))

    print(f'index {threshold} done')

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent


modelname = 't5-base-msmarco-epoch-5'

root_file_name = f'{root_dir}/{modelname}.lz4'
if not os.path.exists(root_file_name):
    os.system(f'cp /nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4 {root_dir}/')
cache_file = Lz4PickleCache(f'{root_dir}/{modelname}.lz4')

pert_file = f'{nfs_dir}/bm25/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30/5, 60/5, 90/5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)


for i, threshold in enumerate(percent):
    # if i == 0:
    #     continue
    create_index(i, threshold, cache_file, modelname, qual_signal="prob")



