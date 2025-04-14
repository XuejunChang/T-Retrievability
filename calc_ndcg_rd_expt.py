import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

import pyterrier as pt
if not pt.java.started():
    pt.java.init()

pt.tqdm.pandas()
import ir_datasets
import ir_measures
from ir_measures import * # imports all supported measures, e.g., AP, nDCG, RR, P
import statistics

import os
import sys
os.environ["PIP_ROOT_USER_ACTION"] = "ignore"
import glob
from itertools import islice

# dataset_name = 'msmarco-passage'
# dataset = pt.get_dataset(f'irds:{dataset_name}')

dataset = pt.get_dataset(f'irds:msmarco-passage')
# df_dataset = pd.DataFrame(dataset.get_corpus_iter(verbose=True))
eval_dev = pt.get_dataset(f'irds:msmarco-passage/dev')
dev_topics = eval_dev.get_topics()
# qrels = eval_dev.get_qrels()

dev_eval = ir_datasets.load("msmarco-passage/dev")
# topics = pd.DataFrame(eval.queries_iter())
dev_qrels = pd.DataFrame(dev_eval.qrels_iter())



dl19 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019')
dl19_topics = dl19.get_topics()
# dl19_qrels = dl19.get_qrels()

dl20 = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')
dl20_topics = dl20.get_topics()
# dl20_qrels = dl20.get_qrels()

dl19_eval = ir_datasets.load("msmarco-passage/trec-dl-2019")
# topics = pd.DataFrame(eval.queries_iter())
dl19_qrels = pd.DataFrame(dl19_eval.qrels_iter())


dl20_eval = ir_datasets.load("msmarco-passage/trec-dl-2020")
# topics = pd.DataFrame(eval.queries_iter())
dl20_qrels = pd.DataFrame(dl20_eval.qrels_iter())

dl1920_topics = pd.concat([dl19_topics, dl20_topics], ignore_index=True)
dl1920_qrels = pd.concat([dl19_qrels, dl20_qrels], ignore_index=True)


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

def Gini(arr):
    arr = np.asarray(arr)
    sorted_arr = np.sort(arr)
    cumarr = np.cumsum(sorted_arr)
    sumarr = cumarr[-1]
    n = len(arr)
    gini = (n + 1 - 2 * np.sum(cumarr) / sumarr) / n
    return gini

def calc_eff(df,qrels):
    df['query_id'] = df['qid'].astype(str)
    df['doc_id'] = df['docno'].astype(str)
    m = ir_measures.calc_aggregate([nDCG @ 10, RR], qrels, df)

    return m[nDCG @ 10], m[RR]

def calc_r_gini_stats(df, topics, doc_rscores_pkl, filter=False):
    if not os.path.exists(doc_rscores_pkl):
        if filter:
            print('filtering')
            qids_to_keep = topics['qid'].to_list()
            mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
            df = df[mask]

        print('groupby docno ...')
        grouped_docs = df.groupby("docno")[['r_score']].sum().reset_index()

        print(f'save to {doc_rscores_pkl}')
        grouped_docs.to_pickle(doc_rscores_pkl, index=False)
    else:
        grouped_docs = pd.read_pickle(doc_rscores_pkl)
        print(f'loaded from {doc_rscores_pkl}')

    scores = grouped_docs['r_score'].to_list()
    print('calc gini')
    gini_value = Gini(scores)
    
    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    avg_r_score = grouped_docs['r_score'].mean()
    min_r_score = grouped_docs['r_score'].min()
    max_r_score = grouped_docs['r_score'].max()

    return mean, std_dev, gini_value, avg_r_score, min_r_score, max_r_score

def group_exec(run_model, thres, group_thres, grouped, topicname, qrels, retrieve_topic_name, filter=True, delcache=False):
    for modelname in run_model:
        os.makedirs(f'{data_dir}/{modelname}', exist_ok=True)
        this_model_res = []
        for threshold in thres:
            print(f'start  {modelname} --> {topicname} -- group_threshold  {group_thres} --> threshold {threshold}')
            retrieved_csv = f'{data_dir}/{modelname}/df_{modelname}_{retrieve_topic_name}_{threshold}.csv'
            rscore_pkl = f'{data_dir}/{modelname}/df_{modelname}_{retrieve_topic_name}_rscore_{threshold}.pkl' 
            doc_rscores_pkl = None
            subgroups_pkl = f'{data_dir}/{modelname}/{modelname}_{topicname}_{group_thres}_T{threshold}.pkl'
            group_threhold_pkl = f'{data_dir}/{modelname}/{modelname}_{topicname}_{group_thres}.pkl'

            if delcache:
                df = pd.read_csv(retrieved_csv, index_col=0).reset_index()
                df['r_score'] = df['rank'].progress_apply(lambda x: 100 / np.log(x + 2))
                print(f'saving {rscore_pkl}')
                df.to_pickle(rscore_pkl)
                print(f'saved')
                            
            if delcache and os.path.exists(subgroups_pkl):
                os.remove(subgroups_pkl)
                print(f'old {subgroups_pkl} removed')
                
            if delcache and os.path.exists(group_threhold_pkl):
                os.remove(group_threhold_pkl)
                print(f'old {group_threhold_pkl} removed')

            print(f'loading {rscore_pkl}')
            df = pd.read_pickle(rscore_pkl)
            res = []
            for group_id, queries_df in grouped:
                print(f'Each group computation: {modelname} ----> {topicname} --> group_threshold  {group_thres} ---> group_id={group_id} ----> threshold {threshold}')
                doc_rscores_pkl = f'{data_dir}/{modelname}/groups/{modelname}_{topicname}_{group_thres}_G{group_id}_T{threshold}.pkl' # subgroup computation
                if delcache:
                    os.remove(doc_rscores_pkl)
                    print(f'old {doc_rscores_pkl} removed')
                mean, std_dev, gini_value, avg_r_score, min_r_score, max_r_score = calc_r_gini_stats(df, queries_df, doc_rscores_pkl, filter=filter)
                nDCG10, rr = calc_eff(df, qrels) if not filter else (None, None)
                group_res = [modelname, threshold, group_thres, group_id, mean, std_dev, gini_value, avg_r_score, min_r_score, max_r_score, nDCG10, rr]
                res.append(group_res)
            print(f'all subgroups for this pruning threshold {modelname} --> {topicname} --> nubmer of groups:{group_thres} --> threshold {threshold}')
            df_threshold = pd.DataFrame(res, columns=['modelname', 'threshold','group_thres', 'group_id', 'mean', 'std', 'gini', 'avg_r_score', 'min_r_score', 'max_r_score', 'nDCG@10', 'RR'])

            print(f'saving {subgroups_pkl}')
            df_threshold.to_pickle(subgroups_pkl)
            print('saved')
            print(f'Computation within each subgroup {modelname} --> {topicname} --> nubmer of groups:{group_thres} --> threshold {threshold}')
            ginis = df_threshold['gini']
            min_gini, mean_gini, max_gini = ginis.min(), ginis.mean(), ginis.max()
            avg_localiszed_gini = Gini(df_threshold['avg_r_score'].to_list())
            min_localiszed_gini = Gini(df_threshold['min_r_score'].to_list())
            max_localiszed_gini = Gini(df_threshold['max_r_score'].to_list())
            nDCG10 = df_threshold['nDCG@10'].mean()
            rr = df_threshold['RR'].mean()
            this_model_res.append([modelname, threshold, group_thres, min_gini, mean_gini, max_gini, avg_localiszed_gini, min_localiszed_gini, max_localiszed_gini, nDCG10, rr])

        print(f'this group for all pruning thresholds: {modelname} --> {topicname} --> nubmer of groups:{group_thres}')
        res_df = pd.DataFrame(this_model_res, columns=['modelname', 'threshold', 'group_thres', 'min_gini', 'mean_gini', 'max_gini', 'avg_localiszed_gini', 'min_localiszed_gini', 'max_localiszed_gini', 'nDCG@10', 'RR'])
        res_df = res_df.round(4)
        res_df = res_df.sort_values(by='group_thres')                        
        print(f'saving {group_threhold_pkl}')
        res_df.to_pickle(group_threhold_pkl)
        os.system(f'cp -r {group_threhold_pkl} {prg_dir}/results/')
        print(f'copied {group_threhold_pkl} into {prg_dir}/results/')


# ['bm25', 'bm25_monot5', 'splade'', 'tctcolbert','bm25_tctcolbert']:
# ['bm25', 'bm25_monot5', 'rtr_splade', 'tctcolbert','bm25_tctcolbert']:
# data_dir = f'/nfs/resources/cxj/retrievability-bias'
# data_dir = f'/nfs/datasets/cxj/retrievability-bias-from_resources_ok'

prg_dir = '/nfs/primary/retrievability-bias/results'
os.makedirs(prg_dir, exist_ok=True)
# root_dir = '/root/retrievability-bias'
# os.makedirs(f'{root_dir}/allresults', exist_ok=True)
data_dir = sys.argv[1]
run_model = sys.argv[2:]
"""
dev grouping
"""
num_clusters = [500, 1000, 2000, 5000, 10000]
topicname = 'dev_grps'
retrieve_topic_name  = 'dev'
qrels = dev_qrels
for group_thres in num_clusters:
    print(f'For {group_thres} groups')
    csv = f'{prg_dir}/results/clustered_dev_queries_by_{group_thres}.csv'
    topics_sampled = pd.read_csv(csv, index_col=0).reset_index()
    grouped = topics_sampled.groupby('cluster')
    thres = [0, 30, 60, 90]
    group_exec(run_model, thres, group_thres, grouped, topicname, qrels, retrieve_topic_name, filter=True, delcache=True)

# """
# dev all
# """
# dev_topics['cluster'] = 0
# grouped = dev_topics.groupby('cluster')
# topicname = 'dev_all'
# retrieve_topic_name  = 'dev'
# qrels = dev_qrels
# thres = [0]
# group_thres = 1
# group_exec(run_model, thres, group_thres, grouped, topicname, qrels, retrieve_topic_name, filter=True, delcache=True)

# """
# dl1920 all
# """
# dl1920_topics['cluster'] = 0
# grouped = dl1920_topics.groupby('cluster')
# topicname = 'dl1920_all'
# retrieve_topic_name  = 'dl1920'
# qrels = dl1920_qrels
# thres = [0]
# group_thres = 1
# group_exec(run_model, thres, group_thres, grouped, topicname, qrels, retrieve_topic_name, filter=True, delcache=True)
