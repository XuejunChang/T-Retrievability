import numpy as np
from ir_measures import *
import pandas as pd
import os,time,sys
import subprocess
import config
from pathlib import Path

def old_gini(v, intv=11):
    v = np.array(v)
    if np.all(v == 0):
        return 0.0
    bins = np.linspace(0., 100., intv)
    total = float(np.sum(v))
    yvals = [0]
    for b in bins[1:]:
        bin_vals = v[v <= np.percentile(v, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100.0
        yvals.append(bin_fraction)
    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # pe_area = np.trapezoid(bins, x=bins)
    
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    # lorenz_area = np.trapezoid(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)
    return gini_val
    
def raw_gini(arr):
    # start = time.time()
    arr = np.array(arr)
    if np.all(arr == 0):
        return 0.0
    n = len(arr)
    x_avg = np.mean(arr)
    sum = 0.0
    for i in range(n):
        for j in range(n):
            sum = sum + np.abs(arr[i]-arr[j])
    denom = 2 * n * n * x_avg
    # end = time.time()
    # print(f'total time: {end-start} s, {(end-start)/60} min')
    return sum/denom

# def Gini(x):
#     # Ensure the array is sorted
#     sorted_x = np.sort(x)
#     n = len(x)
#     # Calculate the Lorenz curve cumulative values
#     cumulative_x = np.cumsum(sorted_x, dtype=float)
#     # Calculate the Gini coefficient using the Lorenz curve
#     gini = (n + 1 - 2 * np.sum(cumulative_x) / cumulative_x[-1]) / n
#     print(f'gini: {gini}')
#     return gini

# def Gini(arr):
#     arr = np.abs(arr)
#     sum_i = 0.0
#     for x_i in arr:
#         sum_j = 0.0
#         for x_j in arr:
#             sum_j = sum_j + np.abs(x_i-x_j)
#         sum_i = sum_i + sum_j
#     denom = 2 * len(arr) + np.sum(arr)
#     return sum_i/denom

def curve_gini(arr):
    arr = np.array(arr)
    if np.all(arr == 0):
        return 0.0
    sorted_arr = np.sort(arr)
    cumarr = np.cumsum(sorted_arr, dtype=float)
    sumarr = cumarr[-1]
    n = len(arr)
    gini = (n + 1 - 2 * np.sum(cumarr) / sumarr) / n
    return gini

# def calc_metrics(df,qrels):
#     df['query_id'] = df['qid'].astype(str)
#     df['doc_id'] = df['docno'].astype(str)
#     m = calc_aggregate([nDCG@10, AP(rel=2), RR], qrels, df)
#     # print(f"nDCG@10: {m[nDCG @ 10]}, AP: {m[AP]}, RR: {m[RR]}")
#     print(m)
#     df['nDCG@10'], df['AP(rel=2)'], df['RR'] = m[nDCG@10], m[AP(rel=2)], m[RR]
#     return df
def get_trec_qrels(df, qrels_res_path):
    if  not os.path.exists(qrels_res_path):
        print(f'saving into {qrels_res_path}')
        result = pd.DataFrame()
        result['query_id'] = df['qid']
        result['Q0'] = 0
        result['doc_id'] = df['docno']
        result['relevance'] = df['label']

        result.to_csv(qrels_res_path, sep=' ', index=False, header=False)
        print(f'saved')

    return qrels_res_path

def save_trec_res(result_csv,run_name, data_dir):
    trec_res_path = f'{data_dir}/{Path(result_csv).stem}.res'
    if not os.path.exists(trec_res_path):
        df = pd.read_csv(result_csv, index_col=0).reset_index()
        print(f'saving into {trec_res_path}')
        result = pd.DataFrame()
        result['query_id'] = df['qid']
        result['Q0'] = 'Q0'
        result['doc_id'] = df['docid']
        result['rank'] = df['rank']
        result['score'] = df['score']
        result['run_name'] = run_name

        result.to_csv(trec_res_path, sep=' ', index=False, header=False)
        print(f'done')

    return trec_res_path

def save_retrieved_docs_measures(result_csv, trec_res_path, data_dir):
    result_measures_path = f'{data_dir}/{Path(result_csv).stem}_measures.csv'
    if not os.path.exists(result_measures_path):
        df = pd.read_csv(result_csv, index_col=0).reset_index()
        print(f'calculating metrics')
        qrels_res_path = f'{data_dir}/qrels_dev.res'
        trec_qrels_path = get_trec_qrels(config.qrels, qrels_res_path)
        metrics_dict = cal_metrics(trec_qrels_path, trec_res_path)
        for items in metrics_dict.items():
            df[items[0]] = items[1]

        print(f'saving into {result_measures_path}')
        df.to_csv(result_measures_path, index=False)
        print(f'done')

def cal_metrics(trec_qrels_path, trec_res_path):
    # ensure that cp /mnt/primary/exposure-fairness/trec_eval /usr/local/bin/
    
    # all_metrics = [
    #     "map", "set_map", "set_P", "set_recall", "set_F", "Rprec", "bpref", "recip_rank",
    #     "ndcg", "ndcg_cut.5", "ndcg_cut.10", "ndcg_cut.20",
    #     "P.5", "P.10", "P.20", "P",
    #     "recall.5", "recall.10", "recall.20", "recall",
    #     "iprec_at_recall.0.0", "iprec_at_recall.0.1", "iprec_at_recall.0.2",
    #     "iprec_at_recall.0.3", "iprec_at_recall.0.4", "iprec_at_recall.0.5",
    #     "iprec_at_recall.0.6", "iprec_at_recall.0.7", "iprec_at_recall.0.8",
    #     "iprec_at_recall.0.9", "iprec_at_recall.1.0",
    #     "F.5", "F.10", "F.20", "F", "relstring"
    # ]

    metrics = ["ndcg_cut.10", "map", "recip_rank", "P.10"]
    args = []
    for metric in metrics:
        args.append('-m')
        args.append(metric)
    cmd = ["trec_eval"] + args + [trec_qrels_path, trec_res_path]

    # cmd = ["trec_eval"] + [trec_qrels_path, trec_res_path]
    # cmd = ["trec_eval"] + ["-q"] + [trec_qrels_path, trec_res_path]

    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)

    metric_dict = {}
    for line in result.stdout.splitlines():
        arr = line.split()
        metric_dict[arr[0]] = float(arr[-1])

    print(metric_dict)
    return metric_dict


# model_name = sys.argv[1]
if __name__ == '__main__':
    trec_qrels_path = '/nfs/datasets/cxj/exposure-fairness/v1/qrels_dev.res'
    trec_res_path = '/nfs/datasets/cxj/exposure-fairness/v1/bm25_tctcolbert_100.res'
    trec_qrels_path = f'{config.data_dir}/qrels_dev.res'
    trec_res_path = f'{config.data_dir}/bm25_100.res'

    result = cal_metrics(trec_qrels_path, trec_res_path)


