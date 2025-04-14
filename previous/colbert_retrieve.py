import warnings

from initmodel.bm25_contriever_retrieve import index_path

warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

import os
import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', False)

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/datasets/cxj/retrievability-bias'
if not os.path.exists(nfs_dir):
    os.makedirs(nfs_dir)

from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory

def create_colbert_indexes(index_path):
    print(f"start indexing into {index_path}")
    index_dir = "/".join(index_path.split('/')[:-1])
    index_name = index_path.split('/')[-1]
    # index_path = "msmarco-passage-colbert-index"
    indexer = ColBERTIndexer(CHECKPOINT, index_dir, index_name, chunksize=8, gpu=True)
    indexer.index(dataset.get_corpus_iter(verbose=True))

# from pyterrier_caching import RetrieverCache
# def get_cached_contriever(index_path, cache_dir):
#     if not os.path.exists(cache_dir):
#         print('start contriever retrieval caching')
#         colbert = ColBERTFactory(CHECKPOINT, f"{root_dir}/", f"{colbert_index}")
#         # Create pipeline
#         pipeline = bm25 >> pt.text.get_text(dataset, 'text') >> colbert.text_scorer()
#         cached_contriever = RetrieverCache(cache_dir, retriever)
#     else:
#         cached_contriever = RetrieverCache(cache_dir)
#
#     return cached_contriever

if __name__ == '__main__':
    # CHECKPOINT = "/nfs/primary/data/llm/colbert_v2/colbert.dnn"
    # index_path = '/nfs/datasets/cxj/retrievability-bias/colbert/msmarco-passage-colbert-index'
    CHECKPOINT = "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
    pytcolbert = ColBERTFactory(CHECKPOINT, f"{nfs_dir}/colbert", "msmarco-passage-colbert-index", faiss_partitions=100)
    dense_e2e = pytcolbert.end_to_end() % 10
    result = dense_e2e.transform(topics[:2])
    # print(result.head(5))

    # dense_e2e = pytcolbert.end_to_end(start=start, end=end)
    # pytcolbert.end_to_end(start=start, end=end)

    # print('start colbert retrieval')
    # topics.sort_values(by='qid')
    #
    # br = pt.terrier.Retriever(index, wmodel='BM25',verbose=True) % 100
    # pipe = br >> pt.text.get_text(dataset,'text')


    #
    # csv = f'results_colbert.csv'
    # result.to_csv(f'{root_dir}/{csv}')
    # os.system(f'cp -r {root_dir}/{csv} {nfs_dir}/')
    # print(f'copied {root_dir}/{csv} into {nfs_dir}')
