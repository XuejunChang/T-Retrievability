import numpy as np
from tqdm import tqdm
tqdm.pandas()

import pandas as pd
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


# def sum_doc_rscores_each_group(df, grouped_queries_df):
#     print('groupby docno ...')
#     qids_to_keep = grouped_queries_df['qid'].to_list()
#     mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
#     df = df[mask]

#     sum_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()
#     return sum_doc_rscores


# def calc_metrics(sum_doc_rscores):
#     print('calc metrics')
#     scores = sum_doc_rscores['r_score'].to_list()
#     mean = statistics.mean(scores)
#     std = statistics.stdev(scores)
    
#     avg_rscore = sum_doc_rscores['r_score'].mean()
#     min_rscore = sum_doc_rscores['r_score'].min()
#     max_rscore = sum_doc_rscores['r_score'].max()

#     return mean, std, group_gini, avg_rscore, min_rscore, max_rscore

# args: [version] [delcache] [run_model ... ]
version = sys.argv[1]
data_dir = f'{config.data_dir}/{version}'
os.makedirs(data_dir, exist_ok=True)

method = sys.argv[2]
run_model = sys.argv[3:]

num_clusters = [500, 1000, 2000, 5000, 10000]
# one-pass computation of retrievability scores for the retrieved documents
def cacl_res_rscore(run_model):
    for modelname in run_model:
        print(f'evaluate {modelname} by {granu}')    
        rscore_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'    
        if not os.path.exists(rscore_csv):
            result_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}.csv'
            df = pd.read_csv(result_csv, index_col=0).reset_index()
            print('calc r_score of each document of the retrieved docs ...')
            df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))
            print(f'saving into {rscore_csv}')
            df.to_csv(rscore_csv, index=False)
            print(f'done')


def add_clusterid_to_res_df(num_clusters, run_model):
    for granu in num_clusters:
        query_csv_path = f'{config.prog_dir}/grouped_queries/clustered_dev_queries_by_{granu}.csv'
        cluster_q_df = pd.read_csv(query_csv_path, index_col=0).reset_index()
    
        for modelname in run_model:
            print(f'processing granularity {granu} --> {modelname}')
            result_csv_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'
            res_df = pd.read_csv(result_csv_path, index_col=0).reset_index()
    
            merged_df = res_df.merge(cluster_q_df, on='qid', how='left')
            merged_file_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}.csv'
            if os.path.exists(merged_file_path):
                os.remove(merged_file_path)
            print(f'saving {merged_file_path}')
            merged_df.to_csv(merged_file_path, index=False)
            print('done')

def calc_topical_gini(num_clusters, run_model):
    models_granularities = []
    for granu in num_clusters:
        for modelname in run_model:
            topical_res_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}.csv'
            res_df = pd.read_csv(topical_res_path, index_col=0).reset_index()
            grouped_df = res_df.groupby('cluster')

            res = []
            for group_id, df in grouped_df:            
                sum_rscores = df.groupby("docno")[['r_score']].sum().reset_index()
                
                print('calc group_gini')
                group_gini = gini.compute_gini(sum_rscores['r_score'].to_dict())
                print(f'group_gini: {group_gini}')
                
                res.append([modelname, granu, group_id, group_gini])

            df_granularity_wise = pd.DataFrame(res, columns=['modelname', 'granu', 'group_id','group_gini'])
            ginis = df_granularity_wise['group_gini']
            models_granularities.append([modelname, granu, ginis.min(), ginis.mean(), ginis.max()])

    print(f'save the statistics of the model {modelname}')
    result_csv = pd.DataFrame(models_granularities, columns=['modelname', 'granu', 'min_gini', 'mean_gini', 'max_gini'])
    result_csv_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_models_granularities.csv'
    if os.path.exists(result_csv_path):
        os.remove(result_csv_path)
                    
    print(f'saving into {result_csv_path}')
    result_csv.to_csv(result_csv_path, index=False)
    print('done')

def save_topical_trec_res(num_clusters, run_model):
    for granu in num_clusters:
        for modelname in run_model:
            print(f'processing granularity {granu} --> {modelname}')
            result_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}.csv'
            run_name = f'{modelname} res with {granu} clusters'
            fair_utils.save_topical_trec_res(result_csv,run_name, data_dir)


if __name__ == "__main__":

    if method =='add_clusterid':
        add_clusterid_to_res_df(num_clusters, run_model)

    if method =='save_trec':
        save_topical_trec_res(num_clusters, run_model)

    if method =='calc_gini':
        calc_topical_gini(num_clusters, run_model)
        
        
    
    
    



