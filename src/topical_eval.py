import numpy as np
from tqdm import tqdm

import convert
tqdm.pandas()
import pandas as pd
import fair_utils
import os
import config
import argparse

def mapping_to_cluster(trec_file_path, clustered_queries_path):
    ret_df = convert.convert_res2docdf(trec_file_path, config.trec_res_columns)
    print('Computing retrievability for each document')
    ret_df['r_score'] = ret_df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))

    cluster_q_df = pd.read_csv(clustered_queries_path, index_col=0).reset_index()
    print(f'Mapping cluster id for {trec_file_path}')
    topical_df = ret_df.merge(cluster_q_df, on='qid', how='left')

    return topical_df

def calc_topical_gini(topical_df):
    print(f'Computing aggregated topical gini')
    grouped_df = topical_df.groupby('cluster')
    res = []
    for group_id, df in grouped_df:
        sum_rscores = df.groupby("docno")['r_score'].sum().reset_index()
        print('calc group_gini')
        group_gini = fair_utils.compute_gini(sum_rscores['r_score'].to_dict())
        print(f'group_gini: {group_gini}')
        res.append(group_gini)

    return min(res), sum(res)/len(res), max(res)


# trec_file_path = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}'
# clustered_queries_path = f'{config.prog_dir}/grouped_queries/clustered_dev_queries_by_{granu}_{kmeans_vec}.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accept one model name, a vector representation for K-means and the number of groups")
    parser.add_argument('--trec_file_path', help='The path of the trec file')
    parser.add_argument('--clustered_queries_path', help='The path of the clustered queries')

    args = parser.parse_args()
    trec_file_path = args.trecfile
    clustered_queries_path = args.clustered_queries_path

    topical_df = mapping_to_cluster(trec_file_path, clustered_queries_path)
    min_gini, mean_gini, max_gini = calc_topical_gini(topical_df)
    print(f'min_gini: {min_gini}, mean_gini: {mean_gini}, max_gini: {max_gini}')

