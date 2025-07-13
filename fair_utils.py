from wsgiref.util import request_uri

import numpy as np
from ir_measures import *
import pandas as pd
import os, time, sys
import subprocess
import config
from pathlib import Path

def get_trec_queries(df, query_res_path):
    if not os.path.exists(query_res_path):
        print(f'saving into {query_res_path}')
        df.to_csv(query_res_path, sep=' ', index=False, header=False)
        print(f'saved')
    else:
        print(f'found {query_res_path}')

    return query_res_path

def get_trec_qrels(df, qrels_res_path):
    if os.path.exists(qrels_res_path):
        print(f'found {qrels_res_path}')
    else:
        result = pd.DataFrame()
        result['query_id'] = df['qid']
        result['Q0'] = 0
        result['doc_id'] = df['docid']
        result['relevance'] = df['label']
        
        print(f'saving into {qrels_res_path}')
        result.to_csv(qrels_res_path, sep=' ', index=False, header=False)
        print(f'saved')

    return qrels_res_path

def save_trec_res(df, res_file_path, run_name):
    if os.path.exists(res_file_path):
        os.remove(res_file_path)
        print(f'removed {res_file_path}')

    result = pd.DataFrame()
    result['query_id'] = df['qid']
    result['Q0'] = 'Q0'
    result['doc_id'] = df['docid']
    result['rank'] = df['rank']
    result['score'] = df['score']
    result['run_name'] = run_name

    print(f'saving into {res_file_path}')
    result.to_csv(res_file_path, sep=' ', index=False, header=False)
    print(f'done')

def convert_res2df(res_file_path):
        print(f'converting {res_file_path} into a dataframe')
        df = pd.read_csv(res_file_path,sep=r"\s+", names=["qid", "Q0", "docid", "rank", "score", "run"], dtype={"qid": str, "docid": str})
        df['docno'] = df['docid']
        df = df.merge(config.topics, how='left', on='qid')
        print('done')
        return df
    
def cal_metrics(qrels_path, docs_path):
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
    cmd = ["trec_eval"] + args + [qrels_path, docs_path]
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
    qrels_path = '/nfs/datasets/cxj/exposure-fairness/v1/qrels_dev.res'
    docs_path = '/nfs/datasets/cxj/exposure-fairness/v1/bm25_tctcolbert_100.res'
    qrels_path = f'{config.data_dir}/qrels_dev.res'
    docs_path = f'{config.data_dir}/bm25_100.res'

    result = cal_metrics(qrels_path, docs_path)
