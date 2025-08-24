import sys

import numpy as np
from tqdm import tqdm

import convert
tqdm.pandas()
import pandas as pd
import fair_utils
import os
import config
import argparse


def calc_topical_gini(trec_df, cluster_q_df):
    print('Computing retrievability for each document')
    trec_df['r_score'] = trec_df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))
    print(f'Mapping cluster id for the trec file')
    topical_df = trec_df.merge(cluster_q_df, on='qid', how='left')

    print(f'Computing aggregated topical gini')
    grouped_df = topical_df.groupby('cluster')
    res = []
    for group_id, df in tqdm(grouped_df):
        sum_rscores = df.groupby("docno")['r_score'].sum().reset_index()
        group_gini = fair_utils.compute_gini(sum_rscores['r_score'].to_dict())
        res.append(group_gini)

    return min(res), sum(res)/len(res), max(res)


# trec_file_path = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}'
# clustered_queries_path = f'{config.proj_dir}/grouped_queries/clustered_dev_queries_by_{granu}_{kmeans_vec}.csv'

retrieved_res_file = sys.argv[1]
clustered_queries = sys.argv[2]
if __name__ == "__main__":
    retrieved_res_path = f'{config.data_dir}/{retrieved_res_file}'
    clustered_queries_path = f'{config.proj_dir}/grouped_queries/{clustered_queries}'

    trec_df = convert.convert_res2docdf(retrieved_res_path, config.trec_res_columns)
    cluster_q_df = pd.read_csv(clustered_queries_path, index_col=0).reset_index()
    cluster_q_df['qid'] = cluster_q_df['qid'].astype(str)

    min_gini, mean_gini, max_gini = calc_topical_gini(trec_df, cluster_q_df)
    print(f'min_gini: {min_gini:.4f}, mean_gini: {mean_gini:.4f}, max_gini: {max_gini:.4f}')

