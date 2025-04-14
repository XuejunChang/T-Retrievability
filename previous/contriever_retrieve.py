import sys
import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()
import faiss
import datetime
import os
import numpy as np
import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', False)
from contriever_model import Contriever

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

work_name = "retrievability-bias"
root_dir = f'/root/{work_name}'
nfs_save = f'/nfs/datasets/cxj/{work_name}'
if not os.path.exists(nfs_save):
    os.makedirs(nfs_save)

import pyterrier_dr

model_name = "facebook/contriever-msmarco"
model = Contriever(model_name)

faiss_index_dir = f"{nfs_save}/contriever/pt_contriever_index"
os.makedirs(faiss_index_dir, exist_ok=True)
index_path = f'{faiss_index_dir}/contriever.flex'

def create_index(index_path):
    index = pyterrier_dr.FlexIndex(index_path)
    idx_pipeline = model >> index
    idx_pipeline.index(dataset.get_corpus_iter(verbose=True))

create_index(index_path)

### end-to-end retrieve
index = faiss.read_index(index_path)
retr_pipeline = model >> index.torch_retriever()
results = retr_pipeline.transform(topics)
csv = f'results_bm25_contriever_e2e_100.csv'
results.to_csv(f'{root_dir}/{csv}')
os.system(f'cp -r {root_dir}/{csv} {nfs_save}/')
print(f'copied {root_dir}/{csv} into {nfs_save}')




bm25 = pt.BatchRetrieve(lexical_index, wmodel='BM25', verbose=True) % 100
results = pipeline(topics)



# res_df = pd.DataFrame(res, columns=["qid", "did", "rank", "distance"])
# print('start saving to csv file')
# csvfile = f'/nfs/datasets/cxj/retrievability-bias/contriever/results_contriever_100_qbatch_{j}.csv'
# if os.path.exists(csvfile):
#     os.remove(csvfile)
#     print(f'{csvfile} deleted')
#
# res_df.to_csv(csvfile, index=False)
# print(f'{csvfile} saved')
# del res_df
# del res


