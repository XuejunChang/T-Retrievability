import numpy as np
from ir_measures import *
import pandas as pd
import os,time,sys
import subprocess
import config
from pathlib import Path

# def calc_metrics(df,qrels):
#     df['query_id'] = df['qid'].astype(str)
#     df['doc_id'] = df['docno'].astype(str)
#     m = calc_aggregate([nDCG@10, AP(rel=2), RR], qrels, df)
#     # print(f"nDCG@10: {m[nDCG @ 10]}, AP: {m[AP]}, RR: {m[RR]}")
#     print(m)
#     df['nDCG@10'], df['AP(rel=2)'], df['RR'] = m[nDCG@10], m[AP(rel=2)], m[RR]
#     return df

def get_trec_queries(df, query_res_path):
    if  not os.path.exists(query_res_path):
        print(f'saving into {query_res_path}')
        df.to_csv(query_res_path, sep=' ', index=False, header=False)
        print(f'saved')

    return query_res_path
    
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


