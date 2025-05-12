import numpy as np
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
import fair_utils
import statistics

import os
import sys
from pathlib import Path
import gini
import config

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

    avg_rscore = sum_doc_rscores['r_score'].mean()
    min_rscore = sum_doc_rscores['r_score'].min()
    max_rscore = sum_doc_rscores['r_score'].max()
    fast_gini = gini.compute_gini(sum_doc_rscores['r_score'].to_dict())

    return mean, std, fast_gini, avg_rscore, min_rscore, max_rscore

# args: [version] [delcache] [run_model ... ]
version = sys.argv[1]
data_dir = f'{config.data_dir}/{version}'
os.makedirs(data_dir, exist_ok=True)

delcache = True if sys.argv[2] == 'True' else False
run_model = sys.argv[3:]

num_clusters = [500, 1000, 2000, 5000, 10000]
topic = 'dev'
results_df = pd.DataFrame()
for modelname in run_model:
    for num_groups in num_clusters:
        print(f'evaluate {modelname} by {num_groups}')
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
            print(f'Each group computation: {modelname} ---->  {num_groups} ---> group_id={group_id}')
            summed_doc_rscores = sum_doc_rscores_each_group(df, grouped_queries_df)
            mean, std, fast_gini, avg_rscore, min_rscore, max_rscore = calc_metrics(summed_doc_rscores)
            this_group_res = [modelname, num_groups, group_id, mean, std, fast_gini, avg_rscore, min_rscore, max_rscore]
            res.append(this_group_res)

        print(f'Put the results of each group into a dataframe for nubmer of groups:{num_groups} for {modelname}')
        df_num_groups = pd.DataFrame(res, columns=['modelname', 'num_groups', 'group_id', 'mean', 'std', 'fast_gini',
                                                   'avg_rscore', 'min_rscore', 'max_rscore'])

        print(f'Aggregate groups for nubmer of groups:{num_groups} for {modelname}')
        ginis = df_num_groups['fast_gini']
        df_num_groups['fast_gini_min'], df_num_groups['fast_gini_mean'], df_num_groups[
            'fast_gini_max'] = ginis.min(), ginis.mean(), ginis.max()

        # avg_localiszed_gini = Gini(df_num_groups['avg_rscore'].to_list())
        # conbine all models for this number of groups
        df_num_groups = df_num_groups.sort_values(by=['modelname', 'num_groups', 'group_id'])

        # combine all number of groups
        results_df = pd.concat([results_df, df_num_groups], ignore_index=True)

results_pkl = f'{data_dir}/results.pkl'
print(f'saving into {results_pkl}')
results_df.to_pickle(results_pkl)
print('saved.')


