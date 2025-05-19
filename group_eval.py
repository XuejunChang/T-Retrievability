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

    print('calc group_gini')
    group_gini = gini.compute_gini(sum_doc_rscores['r_score'].to_dict())
    print('done')

    avg_rscore = sum_doc_rscores['r_score'].mean()
    min_rscore = sum_doc_rscores['r_score'].min()
    max_rscore = sum_doc_rscores['r_score'].max()
    
    return mean, std, group_gini, avg_rscore, min_rscore, max_rscore


# args: [version] [delcache] [run_model ... ]
version = sys.argv[1]
data_dir = f'{config.data_dir}/{version}'
os.makedirs(data_dir, exist_ok=True)

delcache = True if sys.argv[2] == 'True' else False
run_model = sys.argv[3:]

num_clusters = [500, 1000, 2000, 5000, 10000]
topic = 'dev'
for modelname in run_model:
    models_granularities = []
    for grp_granularity in num_clusters:
        print(f'evaluate {modelname} by {grp_granularity}')
        csv = f'{config.prog_dir}/clustered_dev_queries_by_{grp_granularity}.csv'
        topics_sampled = pd.read_csv(csv, index_col=0).reset_index()
        grouped_df = topics_sampled.groupby('cluster')

        rscore_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'
        if delcache:  # all caches refer to files related to r_score.
            Path(rscore_csv).unlink(missing_ok=True)
            print(f'{rscore_csv} removed')

        if os.path.exists(rscore_csv):
            print(f'loading {rscore_csv}')
            df = pd.read_csv(rscore_csv, index_col=0).reset_index()
        else:
            result_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}.csv'
            df = pd.read_csv(result_csv, index_col=0).reset_index()
            print('calc r_score of each document of the retrieved docs ...')
            df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))
            print(f'saving into {rscore_csv}')
            df.to_csv(rscore_csv, index=False)
            print(f'done')

        res = []
        for group_id, grouped_queries_df in grouped_df:
            print(f'Group-wise computation: {modelname} ---->  {grp_granularity} ---> group_id={group_id}')
            summed_doc_rscores = sum_doc_rscores_each_group(df, grouped_queries_df)
            mean, std, group_gini, avg_rscore, min_rscore, max_rscore = calc_metrics(summed_doc_rscores)
            this_group_res = [modelname, grp_granularity, group_id, mean, std, group_gini, avg_rscore, min_rscore,
                              max_rscore]
            res.append(this_group_res)

        print(f'Build granularity-wise df for {grp_granularity} for {modelname}')
        df_granularity_wise = pd.DataFrame(res, columns=['modelname', 'grp_granularity', 'group_id', 'mean', 'std',
                                                         'group_gini', 'avg_rscore', 'min_rscore', 'max_rscore'])
        ginis = df_granularity_wise['group_gini']
        models_granularities.append([modelname, grp_granularity, ginis.min(), ginis.mean(), ginis.max()])

    print(f'add the statistics of the model {modelname} into result_csv')
    result_csv = pd.DataFrame(models_granularities,
                              columns=['modelname', 'grp_granularity', 'min_gini', 'mean_gini', 'max_gini'])
    result_csv_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_models_granularities.csv'
    print(f'saving into {result_csv_path}')
    result_csv.to_csv(result_csv_path, index=False)
    print('done')




