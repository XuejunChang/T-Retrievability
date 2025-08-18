import os
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

prog_dir = '/mnt/primary/exposure-fairness'
# data_dir = '/mnt/datasets/cxj/exposure-fairness'
data_dir = '/home/xuejunchang/OneDrive/PhD/projects/mnt/primary/exposure-fairness-extend/data/'
os.makedirs(data_dir, exist_ok=True)

retrieve_num = 100
num_clusters = [500, 1000, 2000, 5000, 10000]
models = ['bm25', 'splade', 'tctcolbert', 'bm25_tctcolbert', 'bm25_monot5']
kmeans_vec = ['scikit_dense','scikit_tfidf']
trec_res_columns = ['qid', 'Q0', 'docid', 'rank', 'score', 'run_name']

dataset_name = 'msmarco-passage'
topics_name = 'dev'
dataset = pt.get_dataset(f'irds:{dataset_name}')
dev = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = dev.get_topics()
qrels = dev.get_qrels()
qrels_res_dev = f'{data_dir}/qrels_dev.res'