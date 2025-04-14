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
import pyterrier_dr

# def process_in_batches(corpus_iter, batch_size):
#     batch = []
#     for doc in corpus_iter:
#         batch.append(doc)  # Collect documents into the batch
#         if len(batch) >= batch_size:
#             yield batch  # Yield the current batch for processing
#             batch = []  # Reset the batch
#     if batch:
#         yield batch
#
# def create_index(index_path):
#     index = pyterrier_dr.FlexIndex(index_path)
#     idx_pipeline = model >> index
#     for batch in process_in_batches(dataset.get_corpus_iter(verbose=True), batch_size):
#         print(f"Indexing batch of {len(batch)} documents")
#         idx_pipeline.index(batch)

def create_index(index_path):
    index = pyterrier_dr.FlexIndex(index_path)
    idx_pipeline = model >> index
    idx_pipeline.index(dataset.get_corpus_iter(verbose=True))

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

root_dir = '/root/retrievability-bias/'
nfs_save = '/nfs/datasets/cxj/retrievability-bias/'
if not os.path.exists(nfs_save):
    os.makedirs(nfs_save)

batch_size = 64
model_name = "facebook/contriever-msmarco"
model = Contriever(model_name,batch_size=batch_size)

faiss_index_dir = f"{nfs_save}/contriever/pt_contriever_index"
os.makedirs(faiss_index_dir, exist_ok=True)
index_path = f'{faiss_index_dir}/contriever.flex'
# print(f'{datetime.datetime.now()}:start creating index')
# create_index(index_path)


print(f'{datetime.datetime.now()}:start to retrieve')
index = pyterrier_dr.FlexIndex(index_path)
retr_pipeline = model >> index.torch_retriever()

#
q_batch = 1000
columns = None
for i in range(0, topics.shape[0], q_batch):
    print(f'f{datetime.datetime.now()} start query {i}')
    results = retr_pipeline.transform(topics[i:i+q_batch])
    del results['query_vec'] # don't need this column to calc retrievability score
    print(results.columns)

    # print(results.head())
    csv = f'results_contriever_e2e_100_{i}.csv'
    print(f'start to save {csv}')
    results.to_csv(f'{root_dir}/{csv}')
    os.system(f'cp -r {root_dir}/{csv} /nfs/datasets/cxj/retrievability-bias/contriever/result/contriever/result/')
    print(f'copied {root_dir}/{csv} into {nfs_save}/contriever/result')
    del results

    if i == topics.shape[0] -1:
        columns = results.columns

import glob
# head_df = pd.read_csv('/nfs/datasets/cxj/retrievability-bias/contriever/result/results_contriever_e2e_100_101000.csv')
res_df = pd.DataFrame(None, columns=columns)
print(f'f{datetime.datetime.now()} start concating df')
for file in glob.glob(f"/nfs/datasets/cxj/retrievability-bias//contriever/result/*.csv"):
    res_df = pd.concat([res_df, pd.read_csv(file)], ignore_index=True)

res_df.to_csv('/nfs/datasets/cxj/retrievability-bias/results_contriever_raw.csv',index=False)
print('saved df')

