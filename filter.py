import numpy as np, time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import ir_datasets
import ir_measures
import fair_utils
import config
import gini
from ir_measures import *  # imports all supported measures, e.g., AP, nDCG, RR, P
import statistics

import os
import sys
from pathlib import Path

os.environ["PIP_ROOT_USER_ACTION"] = "ignore"
import glob
from itertools import islice

# dataset_name = 'msmarco-passage'
# dataset = pt.get_dataset(f'irds:{dataset_name}')

dataset = pt.get_dataset(f'irds:msmarco-passage')
# df_dataset = pd.DataFrame(dataset.get_corpus_iter(verbose=True))
eval_dev = pt.get_dataset(f'irds:msmarco-passage/dev')
dev_topics = eval_dev.get_topics()
# qrels = eval_dev.get_qrels()

dev_eval = ir_datasets.load("msmarco-passage/dev")
# topics = pd.DataFrame(eval.queries_iter())
dev_qrels = pd.DataFrame(dev_eval.qrels_iter())

def cluster_docs(df, cluster_id, cluster_queries_df):
    print('cluster docs ...')
    qids_to_keep = cluster_queries_df['qid'].to_list()
    mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
    cluster_docs_df = df[mask].copy()
    cluster_docs_df['cluster_id'] = cluster_id

    return cluster_docs_df

def calc_metrics(sum_doc_rscores):
    print('calc metrics')
    scores = sum_doc_rscores['r_score'].to_list()

    mean = statistics.mean(scores)
    std = statistics.stdev(scores)

    # oldgini_10 = fair_utils.old_gini(scores,intv=11)
    # oldgini_100 = fair_utils.old_gini(scores,intv=101)
    # oldgini_1000 = fair_utils.old_gini(scores,intv=1001)
    # raw_gini = fair_utils.raw_gini(scores)
    fast_gini = fair_utils.curve_gini(scores)
    avg_rscore = sum_doc_rscores['r_score'].mean()
    min_rscore = sum_doc_rscores['r_score'].min()
    max_rscore = sum_doc_rscores['r_score'].max()

    # return mean, std, oldgini_10,oldgini_100,oldgini_1000, raw_gini, avg_rscore, min_rscore, max_rscore
    return mean, std, fast_gini, avg_rscore, min_rscore, max_rscore

# args: [version] [run_model ...]
version = sys.argv[1]
data_dir = f'{config.home_data}/{version}'
os.makedirs(data_dir, exist_ok=True)

run_model = sys.argv[2:]

"""
dev grouping
"""
num_clusters = [500, 1000, 2000, 5000, 10000]
grouped_topic = 'dev_grps'
topic = 'dev'
qrels = dev_qrels
results_df = pd.DataFrame()
for modelname in run_model:
    retrieved_csv = f'{data_dir}/df_{modelname}_{topic}_0.csv'
    print(f'loading {retrieved_csv}')
    df = pd.read_csv(retrieved_csv, index_col=0).reset_index()
    for num_groups in num_clusters:
        start = time.time()
        print(f'evaluate {modelname} by {grouped_topic} of {num_groups}')
        csv = f'{config.prog_dir}/clustered_dev_queries_by_{num_groups}.csv'
        topics_sampled = pd.read_csv(csv, index_col=0).reset_index()
        cluster_df = topics_sampled.groupby('cluster') # cluster value means the i-th group

        for cluster_id, cluster_queries_df in cluster_df:
            print(f'Each group computation: {modelname} ----> {grouped_topic} -->  {num_groups} ---> cluster_id={cluster_id}')
            # computation for each group
            cluster_docs_df = cluster_docs(df, cluster_id, cluster_queries_df)
            trec_res_file = f'{data_dir}/{modelname}_{num_groups}_{cluster_id}.res'
            fair_utils.convert_df2Trec(cluster_docs_df, trec_res_file, 'res', run_name=f'{modelname}_{num_groups}')
            rr_map = gini.build_log_reciprocal_rank_map(trec_res_file)
            cluster_gini = gini.compute_gini(rr_map)
            cluster_docs_df['cluster_gini'] = cluster_gini

            results_df = pd.concat([results_df, cluster_docs_df], ignore_index=True)

        end = time.time()
        print(f'{modelname}: total time of number of groups {num_groups}: {(end-start)/60} minutes')

results_csv = f'{data_dir}/results_2.csv'
print(f'saving into {results_csv}')
results_df.to_pickle(results_csv)
print('saved.')

