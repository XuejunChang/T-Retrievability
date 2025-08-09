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

import fair_utils, convert
import os, time
import gini
import config
import argparse

def cacl_res_rscore(run_models, data_dir):
    for modelname in run_models:
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


def calc_topical_gini(num_clusters, run_models, data_dir, km=None):
    start = time.time()
    for modelname in run_models:
        model_granularities = []
        for granu in num_clusters:
            print(f'calc_topical_gini at granularity {granu} --> {modelname}')
            topical_res_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}_{km}.csv'
            res_df = pd.read_csv(topical_res_path, index_col=0).reset_index()
            grouped_df = res_df.groupby('cluster')

            res = []
            for group_id, df in grouped_df:
                sum_rscores = df.groupby("docno")[['r_score']].sum().reset_index()
                print('calc group_gini')
                group_gini = gini.compute_gini(sum_rscores['r_score'].to_dict())
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

    end = time.time()
    print(f'calc_topical_gini total time: {(end - start) / 60} minutes')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accept multiple methods and models")
    parser.add_argument('--methods', nargs='+', help='Which methods to run')
    parser.add_argument('--models', nargs='+', help='Which models to be evaluated')

    args = parser.parse_args()
    run_models = args.models
    num_clusters = config.num_clusters

    if 'transform_trec' in args.methods:
        for modelname in run_models:
            for granu in num_clusters:
                for km in config.kmeans_vec:
                    print(f'transform_topical_trec_res for granularity {granu} --> {modelname}')
                    result_csv = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{granu}_{km}.csv'
                    run_name = f'{modelname} res with {granu} clusters'
                    fair_utils.save_topical_trec_res(result_csv, run_name, config.data_dir)

    if 'add_groupid_into_retrieved_res' in args.methods:
        for modelname in run_models:
            res_path = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}.res'
            print(f'processing {res_path}')
            columns = ["qid", "Q0", "docid", "rank", "score", "run"]
            doc_df = convert.convert_res2docdf(res_path, columns=columns)

            for km in config.kmeans_vec:
                for granu in num_clusters:
                    query_path = f'{config.data_dir}/grouped_queries/clustered_dev_queries_{km}_{granu}.csv'
                    query_df = pd.read_csv(query_path, index_col=0).reset_index()

                    # columns = ['qid', 'Q0', 'docid', 'rank', 'score', 'run', 'cluster']
                    merged_df = doc_df.merge(query_df[['qid', 'cluster']], on='qid', how='left')
                    merged_df = merged_df.sort_values(by=['cluster'])

                    merged_file_path = f'{os.path.splitext(res_path)[0]}_{km}_{granu}_with_groupid.res'
                    run_name = os.path.splitext(merged_file_path)[0].split('/')[-1]
                    convert.convert_dfall2trec(merged_df, merged_file_path)

    if 'cal_topical_gini' in args.methods:
        for modelname in run_models:
            for km in config.kmeans_vec:
                for granu in num_clusters:
                    start = time.time()
                    res_path = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}_{km}_{granu}_with_groupid.res'
                    print(f'processing {res_path}')
                    base_key, min, avg, max = fair_utils.build_log_reciprocal_rank_map(res_path, modelname, granu, km)
                    print(f'{base_key}, min:{min}, avg:{avg}, max:{max}')
                    end = time.time()
                    print(f'Processing takes {(end-start):.2f} seconds')

