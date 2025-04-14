import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

import os
import pandas as pd
pt.tqdm.pandas()

import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

# dataset = pt.get_dataset(f'irds:msmarco-passage')
# eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
# topics = eval_dataset.get_topics()
# qrels = eval_dataset.get_qrels()



nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

def Gini(v):
    v = np.array(v)
    bins = np.linspace(0., 100., 11)
    total = float(np.sum(v))
    yvals = [0]
    for b in bins[1:]:
        bin_vals = v[v <= np.percentile(v, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100.0
        yvals.append(bin_fraction)
    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)
    return gini_val

import statistics
def calc_stats_v2(modelname,df,scoredF, queries_df):
    if not os.path.exists(scoredF):
        qids_to_keep = queries_df['qid'].to_list()
        mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
        df_filtered = df[mask]
        grouped = df_filtered.groupby("docno")[['r_score']].sum().reset_index()
        grouped.to_csv(scoredF,index=False)

    scores = grouped['r_score'].to_list()

    if len(scores) == 0:
        return 0.0, 0.0, 0.0

    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    gini_value = Gini(scores)
    return mean, std_dev, gini_value

sampled_queries = pd.read_csv(f'{nfs_dir}/sampled_dev_queries_50.csv', index_col=0).reset_index()
grouped = sampled_queries.groupby('cluster')

import sys
modelname = sys.argv[1]
pt.tqdm.pandas()

# for group_key, queries_df in islice(grouped,2):
# for modelname in ['bm25', 'bm25_monot5', 'splade', 'colbert', 'bm25_colbert']:
# for threshold in [0, 30, 60, 90]:
for threshold in [30, 60, 90]:

    csv2 = f'/nfs/resources/cxj/retrievability-bias/{modelname}/df_{modelname}_rscore_{threshold}.csv'
    print(f'reading {csv2}')
    df = pd.read_csv(csv2, index_col=0).reset_index()

    res = []
    for cluster_id, queries_df in grouped:
        print(f'start {modelname} ----> threshold {threshold} --> cluster_id = {cluster_id}')
        scoredF = f'{nfs_dir}/{modelname}/groups/{modelname}_T{threshold}_G{cluster_id}.csv'
        mean, std, gini = calc_stats_v2(modelname, df, scoredF, queries_df)
        group_res = [modelname, threshold, cluster_id, mean, std, gini]
        print(group_res)
        res.append(group_res)

    print(f'start creating df per threshold')
    df_threshold = pd.DataFrame(res, columns=['modelname', 'threshold', 'cluster_id', 'mean', 'std', 'gini'])
    res_csv = f'{nfs_dir}/{modelname}/groups/result_T{threshold}_allgroups.csv'
    print(f'saving {res_csv}')
    df_threshold.to_csv(res_csv, index=False)
    print('done')