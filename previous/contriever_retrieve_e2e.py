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
nfs_dir = '/nfs/datasets/cxj/retrievability-bias/'
if not os.path.exists(nfs_dir):
    os.makedirs(nfs_dir)

# print(f'{datetime.datetime.now()}:start creating index')
# create_index(index_path)

nopruned_index = f'{nfs_dir}/contriever/pt_contriever_index/contriever.flex'
if __name__ == '__main__':
    # batch_size = 64
    # model_name = "facebook/contriever-msmarco"
    # # model = Contriever(model_name, batch_size=batch_size)
    #
    #
    # print(f'{datetime.datetime.now()}:start to retrieve')
    # index = pyterrier_dr.FlexIndex(nopruned_index)
    # # retr_pipeline = model >> index.torch_retriever() % 100
    # retr_pipeline = model >> index.np_retriever() % 100
    #
    # print(f'f{datetime.datetime.now()} start to transform topics.shape = {topics.shape}')
    # results = retr_pipeline.transform(topics[:2])
    # print(results.head())
    # print(type(results.columns))
    # # del results['query_vec']  # don't need this column to calc retrievability score
    # #
    # # print(f'f{datetime.datetime.now()} start to save csv')
    # # results.to_csv(f'{nfs_dir}/results_contriever_raw.csv',index=False)
    # # print('saved df')

    model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco')

    index = pyterrier_dr.FlexIndex('./myindextest2.flex')
    # idx_pipeline = model >> index
    # idx_pipeline.index([
    #     {'docno': '0',
    #      'text': 'The Five Find-Outers and Dog, also known as The Five Find-Outers, is a series of children\'s mystery books written by Enid Blyton.'},
    #     {'docno': '1', 'text': 'City is a 1952 science fiction fix-up novel by American writer Clifford D. Simak.'},
    # ])

    retr_pipeline = model >> index.torch_retriever() % 1
    topics = pd.DataFrame([['123','cxj']], columns = ['qid','query'])
    print(topics)
    result = retr_pipeline.transform(topics)
    print(result)


