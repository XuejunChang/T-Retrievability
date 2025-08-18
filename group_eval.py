import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import fair_utils
import os, time
import config
import argparse

def cacl_res_rscore(run_model, data_dir):
    for modelname in run_model:
        print(f'calc retrievability score for {modelname}')
        rscore_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'
        if not os.path.exists(rscore_csv):
            result_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}.csv'
            df = pd.read_csv(result_csv, index_col=0).reset_index()
            print('calc r_score of each document of the retrieved docs ...')
            df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))
            print(f'saving into {rscore_csv}')
            df.to_csv(rscore_csv, index=False)
            print(f'done')
        else:
            print(f'found {rscore_csv}')

def add_clusterid_to_res_df(num_clusters, run_model, data_dir, km=None):
    for granu in num_clusters:
        query_csv_path = f'{config.prog_dir}/grouped_queries/clustered_dev_queries_by_{granu}_{km}.csv'
        cluster_q_df = pd.read_csv(query_csv_path, index_col=0).reset_index()
        for modelname in run_model:
            print(f'add_clusterid_to_res_df {granu} --> {modelname}')
            result_csv_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'
            res_df = pd.read_csv(result_csv_path, index_col=0).reset_index()

            merged_df = res_df.merge(cluster_q_df, on='qid', how='left')
            merged_file_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}_{km}.csv'
            if os.path.exists(merged_file_path):
                os.remove(merged_file_path)
                print(f'{merged_file_path} removed')
            print(f'saving {merged_file_path}')
            merged_df.to_csv(merged_file_path, index=False)
            print('done')

def query_clusterid_to_res_df(num_clusters, run_model, data_dir, km=None):
    for granu in num_clusters:
        query_csv_path = f'{config.prog_dir}/grouped_queries/clustered_dev_queries_by_{granu}_{km}.csv'
        cluster_q_df = pd.read_csv(query_csv_path, index_col=0).reset_index()
        for modelname in run_model:
            print(f'add_clusterid_to_res_df {granu} --> {modelname}')
            result_csv_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_rscore.csv'
            res_df = pd.read_csv(result_csv_path, index_col=0).reset_index()

            merged_df = res_df.merge(cluster_q_df, on='qid', how='left')
            merged_file_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}_{km}.csv'
            if os.path.exists(merged_file_path):
                os.remove(merged_file_path)
                print(f'{merged_file_path} removed')
            print(f'saving {merged_file_path}')
            merged_df.to_csv(merged_file_path, index=False)
            print('done')

def calc_topical_gini(num_clusters, run_model, data_dir, km=None):
    for modelname in run_model:
        model_granularities = []
        for granu in num_clusters:
            print(f'calc_topical_gini at granularity {granu} --> {modelname}')
            topical_res_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}_{km}.csv'
            res_df = pd.read_csv(topical_res_path, index_col=0).reset_index()
            grouped_df = res_df.groupby('cluster')

            res = []
            for group_id, df in grouped_df:
                sum_rscores = df.groupby("docno")['r_score'].sum().reset_index()
                print('calc group_gini')
                group_gini = fair_utils.compute_gini(sum_rscores['r_score'].to_dict())
                print(f'group_gini: {group_gini}')
                res.append([modelname, granu, group_id, group_gini])

            df_granularity_wise = pd.DataFrame(res, columns=['modelname', 'granu', 'group_id', 'group_gini'])
            ginis = df_granularity_wise['group_gini']
            model_granularities.append([modelname, granu, ginis.min(), ginis.mean(), ginis.max()])

        print(f'save the statistics of {modelname}')
        result_csv = pd.DataFrame(model_granularities,
                                  columns=['modelname', 'granu', 'min_gini', 'mean_gini', 'max_gini'])
        result_csv_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_models_granularities_{km}.csv'
        if os.path.exists(result_csv_path):
            os.remove(result_csv_path)
            print(f'{result_csv_path} removed')

        print(f'saving into {result_csv_path}')
        result_csv.to_csv(result_csv_path, index=False)
        print('done')

def transform_topical_trec_res(num_clusters, run_model, data_dir, km=None):
    for modelname in run_model:
        for granu in num_clusters:
            print(f'transform_topical_trec_res for granularity {granu} --> {modelname}')
            result_csv = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}_{km}.csv'
            run_name = f'{modelname} res with {granu} clusters'
            fair_utils.save_topical_trec_res(result_csv, run_name, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accept multiple methods and models")
    parser.add_argument('--methods', nargs='+', help='Which methods to run')
    parser.add_argument('--km', help='Which K-means to run')
    parser.add_argument('--models', nargs='+', help='Which models to be evaluated')
    args = parser.parse_args()

    run_model = args.models
    km = args.km
    print(f'kmeans: {km}')
    num_clusters = config.num_clusters
    if 'add_clusterid' in args.methods:
        cacl_res_rscore(run_model, config.data_dir)
        add_clusterid_to_res_df(num_clusters, run_model, config.data_dir, km=km)

    if 'calc_gini' in args.methods:
        calc_topical_gini(num_clusters, run_model, config.data_dir, km=km)

    if 'transform_trec' in args.methods:
        transform_topical_trec_res(num_clusters, run_model, config.data_dir, km=km)
