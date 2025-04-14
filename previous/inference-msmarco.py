import pyterrier as pt
if not pt.started():
    pt.java.init()

from Lz4PickleCache import *
from T5_Model import *
import os
import time

import warnings
warnings.filterwarnings('ignore')

modelname = "t5-base-msmarco-epoch-5"
root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

# nfs_data_save = f'/nfs/datasets/cxj/{work_name}/efficient'
# if not os.path.exists(nfs_data_save):
#     os.makedirs(nfs_data_save)
    
nfs_data_save = f'/nfs/resources/cxj/retrievability-bias/supervisedT5'
BATCH_SIZE = 16

model_path = f'{root_dir}/{modelname}'
nfs_model_path = f'{nfs_data_save}/{modelname}'
if not os.path.exists(model_path) and os.path.exists(nfs_model_path):
    os.system(f'cp -r {nfs_model_path} {root_dir}/')

filename = f'{root_dir}/{modelname}.lz4'
if os.path.exists(filename):
    os.system(f'rm -f {filename}')

dataset = pt.datasets.get_dataset('irds:msmarco-passage')

cache_file = Lz4PickleCache(filename)
model = T5_Model(model=model_path, batch_size=BATCH_SIZE)
pipeline = model >> cache_file

print(f'start caching to {filename}')

start = time.time()
pipeline.index(dataset.get_corpus_iter(verbose=True), batch_size=BATCH_SIZE)
end  = time.time()
print(f'inferenece time: {end-start} seconds, {(end-start)/60/60} hours.')

os.system(f'cp -r {filename} {nfs_data_save}/')
print('copied.')