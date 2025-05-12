import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
import utils
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


def sum_doc_rscores_each_group(df, grouped_queries_df):
    print('groupby docno ...')
    qids_to_keep = grouped_queries_df['qid'].to_list()
    mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
    df = df[mask]

    sum_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()
    return sum_doc_rscores


def calc_metrics(sum_doc_rscores):
    print('calc metrics')
    scores = sum_doc_rscores['r_score'].to_list()

    mean = statistics.mean(scores)
    std = statistics.stdev(scores)

    # oldgini_10 = utils.old_gini(scores,intv=11)
    # oldgini_100 = utils.old_gini(scores,intv=101)
    # oldgini_1000 = utils.old_gini(scores,intv=1001)
    # raw_gini = utils.raw_gini(scores)
    fast_gini = utils.curve_gini(scores)
    avg_rscore = sum_doc_rscores['r_score'].mean()
    min_rscore = sum_doc_rscores['r_score'].min()
    max_rscore = sum_doc_rscores['r_score'].max()

    # return mean, std, oldgini_10,oldgini_100,oldgini_1000, raw_gini, avg_rscore, min_rscore, max_rscore
    return mean, std, fast_gini, avg_rscore, min_rscore, max_rscore


import config

version = sys.argv[1]
data_dir = f'{config.home_data}/{version}'
os.makedirs(data_dir, exist_ok=True)

delcache = True if sys.argv[2] == 'True' else False
run_model = sys.argv[3:]

"""
dev grouping
"""
num_clusters = [500, 1000, 2000, 5000, 10000]
grouped_topic = 'dev_grps'
topic = 'dev'
qrels = dev_qrels
results_df = pd.DataFrame()
for modelname in run_model:
    for num_groups in num_clusters:
        print(f'evaluate {modelname} by {grouped_topic} of {num_groups}')
        csv = f'{config.prog_dir}/clustered_dev_queries_by_{num_groups}.csv'
        topics_sampled = pd.read_csv(csv, index_col=0).reset_index()
        grouped_df = topics_sampled.groupby('cluster')

        rscore_pkl = f'{data_dir}/df_{modelname}_{topic}_0_rscore.pkl'
        if delcache:  # all caches refer to files related to r_score.
            Path(rscore_pkl).unlink(missing_ok=True)
            print(f'{rscore_pkl} removed')

        if os.path.exists(rscore_pkl):
            print(f'loading {rscore_pkl}')
            df = pd.read_pickle(rscore_pkl)
        else:
            retrieved_csv = f'{data_dir}/df_{modelname}_{topic}_0.csv'
            df = pd.read_csv(retrieved_csv, index_col=0).reset_index()
            print('calc r_score of each document of the retrieved list ...')
            df['r_score'] = df['rank'].progress_apply(lambda x: 1 / np.log(x + 2))
            print(f'saving into {rscore_pkl}')
            df.to_pickle(rscore_pkl)
            print(f'saved')

        res = []
        for group_id, grouped_queries_df in grouped_df:
            print(
                f'Each group computation: {modelname} ----> {grouped_topic} -->  {num_groups} ---> group_id={group_id}')
            # computation for each group
            summed_doc_rscores = sum_doc_rscores_each_group(df, grouped_queries_df)
            mean, std, fast_gini, avg_rscore, min_rscore, max_rscore = calc_metrics(summed_doc_rscores)
            this_group_res = [modelname, num_groups, group_id, mean, std, fast_gini, avg_rscore, min_rscore, max_rscore]
            res.append(this_group_res)

        print(
            f'Put the results of each group into a dataframe for nubmer of groups:{num_groups} for {modelname} {grouped_topic}')
        df_num_groups = pd.DataFrame(res, columns=['modelname', 'num_groups', 'group_id', 'mean', 'std', 'fast_gini',
                                                   'avg_rscore', 'min_rscore', 'max_rscore'])

        print(f'Aggregate groups for nubmer of groups:{num_groups} for  {modelname}  {grouped_topic}')
        # ginis = df_num_groups['oldgini_10']
        # df_num_groups['oldgini_10_min'], df_num_groups['oldgini_10_mean'], df_num_groups['oldgini_10_max'] = ginis.min(), ginis.mean(), ginis.max()

        # ginis = df_num_groups['oldgini_100']
        # df_num_groups['oldgini_100_min'], df_num_groups['oldgini_100_mean'], df_num_groups['oldgini_100_max'] = ginis.min(), ginis.mean(), ginis.max()

        # ginis = df_num_groups['oldgini_1000']
        # df_num_groups['oldgini_1000_min'], df_num_groups['oldgini_1000_mean'], df_num_groups['oldgini_1000_max'] = ginis.min(), ginis.mean(), ginis.max()

        ginis = df_num_groups['fast_gini']
        df_num_groups['fast_gini_min'], df_num_groups['fast_gini_mean'], df_num_groups[
            'fast_gini_max'] = ginis.min(), ginis.mean(), ginis.max()

        # avg_localiszed_gini = Gini(df_num_groups['avg_rscore'].to_list())
        # min_localiszed_gini = Gini(df_num_groups['min_rscore'].to_list())
        # max_localiszed_gini = Gini(df_num_groups['max_rscore'].to_list())

        # conbine all models for this number of groups
        df_num_groups = df_num_groups.sort_values(by=['modelname', 'num_groups', 'group_id'])

        # combine all number of groups
        results_df = pd.concat([results_df, df_num_groups], ignore_index=True)

results_pkl = f'{data_dir}/results.pkl'
print(f'saving into {results_pkl}')
results_df.to_pickle(results_pkl)
print('saved.')

# """
# dev all
# """
# dev_topics['cluster'] = 0
# grouped_df = dev_topics.groupby('cluster')
# grouped_topic = 'dev_all'
# topic  = 'dev'
# qrels = dev_qrels
# thres = [0]
# num_groups = 1
# group_exec(run_model, num_groups, grouped_df, grouped_topic, qrels, topic, delcache=True)

# """
# dl1920 all
# """
# dl1920_topics['cluster'] = 0
# grouped_df = dl1920_topics.groupby('cluster')
# grouped_topic = 'dl1920_all'
# topic  = 'dl1920'
# qrels = dl1920_qrels
# thres = [0]
# num_groups = 1
# group_exec(run_model, num_groups, grouped_df, grouped_topic, qrels, topic, delcache=True)
