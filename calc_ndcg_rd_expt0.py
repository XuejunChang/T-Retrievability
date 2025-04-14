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

def calc_r_gini_stats(df, topics, scoredF, filter=False):
    if not os.path.exists(scoredF):
        if filter:
            print('filtering')
            qids_to_keep = topics['qid'].to_list()
            mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
            df = df[mask]

        print('groupby docno ...')
        grouped = df.groupby("docno")[['r_score']].sum().reset_index()

        print(f'save to {scoredF}')
        grouped.to_csv(scoredF, index=False)
    else:
        grouped = pd.read_csv(scoredF, index_col=0).reset_index()
        print(f'loaded from {scoredF}')

    print('calc mean, std_dev')
    scores = grouped['r_score'].to_list()
    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    print('calc gini')
    gini_value = Gini(scores)

    avg_r_score = grouped['r_score'].mean()
    min_r_score = grouped['r_score'].min()
    max_r_score = grouped['r_score'].max()

    print(mean, std_dev, gini_value, avg_r_score, min_r_score, max_r_score)

    return mean, std_dev, gini_value, avg_r_score, min_r_score, max_r_score

# data_dir = f'/nfs/datasets/cxj/retrievability-bias-from_resources_ok'
data_dir = f'/nfs/resources/cxj/retrievability-bias'
root_dir = '/root/retrievability-bias'
prg_dir = '/nfs/primary/retrievability-bias/results/expt_from_start2'
# os.makedirs(f'{root_dir}/allresults', exist_ok=True)
os.makedirs(f'{data_dir}/allresults', exist_ok=True)
os.makedirs(prg_dir, exist_ok=True)

def group_exec(run_model, thres, grps,eval_topics, grouped, qrels, filter=False):
    # for modelname in ['bm25', 'bm25_monot5', 'rtr_splade', 'tctcolbert','bm25_tctcolbert']:
    # for modelname in ['bm25', 'bm25_monot5', 'splade', 'tctcolbert','bm25_tctcolbert']:
    for modelname in run_model:
        # os.makedirs(f'{root_dir}/{modelname}/groups/new_clustered/', exist_ok=True)
        os.makedirs(f'{data_dir}/{modelname}/groups/new_clustered/', exist_ok=True)
        this_model_res = []
        for threshold in thres:
            """
            Calc retrievability score for each doc
            """
            print(f'start  {modelname} --> {eval_topics} --number of groups {grps} --> threshold {threshold}')
            # origin_topics = eval_topics.split('_')[0]
            origin_topics = 'dev_topics'
            rscore_csv = f'{data_dir}/{modelname}/df_{modelname}_{origin_topics}_rscore_{threshold}_r100.csv' # use 100 / np.log(x + 2)
            if os.path.exists(rscore_csv):
                print(f'loading {rscore_csv}')
                df = pd.read_csv(rscore_csv, index_col=0).reset_index()
            else:
                csv = f'{data_dir}/{modelname}/df_{modelname}_{origin_topics}_{threshold}.csv'
                df = pd.read_csv(csv, index_col=0).reset_index()
                df['r_score'] = df['rank'].progress_apply(lambda x: 100 / np.log(x + 2))
                print(f'saving {rscore_csv}')
                df.to_csv(rscore_csv, index=False)
                print(f'done')

            """
            each group computation
            """
            res = []
            for group_id, queries_df in grouped:
                print(f'Each group computation: {modelname} ----> {eval_topics} --> number of groups {grps} ---> group_id={group_id} ----> threshold {threshold}')
                scoredF = f'{data_dir}/{modelname}/groups/{modelname}_{eval_topics}_{grps}_G{group_id}_T{threshold}_r100.csv' # each group computation
                mean, std_dev, gini_value, avg_r_score, min_r_score, max_r_score = calc_r_gini_stats(df, queries_df, scoredF, filter=filter)
                nDCG10, rr = calc_eff(df, qrels) if not filter else (None, None)
                group_res = [modelname, threshold, grps, group_id, mean, std_dev, gini_value, avg_r_score, min_r_score, max_r_score, nDCG10, rr]
                res.append(group_res)

            """
            Form a csv file
            """
            print(f'merge into a dataframe for {modelname} ----> {eval_topics} ---> nubmer of groups:{grps} ----> threshold {threshold}')
            df_threshold = pd.DataFrame(res, columns=['modelname', 'threshold','grps', 'group_id', 'mean', 'std', 'gini', 'avg_r_score', 'min_r_score', 'max_r_score', 'nDCG@10', 'RR'])
            res_csv = f'{data_dir}/{modelname}/groups/new_clustered/result_{modelname}_{eval_topics}_{grps}_T{threshold}_avg_rscore_r100_new_gini.csv'
            if os.path.exists(res_csv):
                os.remove(res_csv)
                print(f'old {res_csv} removed')
                            
            print(f'saving {res_csv}')
            df_threshold.to_csv(res_csv, index=False)
            print('done')

            """
            Computation over groups
            """
            print(f'Computation over groups {modelname} --> {eval_topics} --> nubmer of groups:{grps} --> threshold {threshold}')
            ginis = df_threshold['gini']
            min_gini, mean_gini, max_gini = ginis.min(), ginis.mean(), ginis.max()
            avg_localiszed_gini = Gini(df_threshold['avg_r_score'].to_list())
            min_localiszed_gini = Gini(df_threshold['min_r_score'].to_list())
            max_localiszed_gini = Gini(df_threshold['max_r_score'].to_list())
            nDCG10 = df_threshold['nDCG@10'].mean()
            rr = df_threshold['RR'].mean()
            this_model_res.append([modelname, threshold, grps, min_gini, mean_gini, max_gini, avg_localiszed_gini, min_localiszed_gini, max_localiszed_gini, nDCG10, rr])

        """
        Form a file with all thresholds.
        """
        print(f'Form a file with all thresholds: {modelname} --> {eval_topics} --> nubmer of groups:{grps}')
        res_df = pd.DataFrame(this_model_res,
                              columns=['modelname', 'threshold', 'grps', 'min_gini', 'mean_gini', 'max_gini', 'avg_localiszed_gini', 'min_localiszed_gini', 'max_localiszed_gini', 'nDCG@10', 'RR'])
        res_df = res_df.round(4)
        res_df = res_df.sort_values(by='grps')

        res_csv = f'{data_dir}/allresults/result_{modelname}_{eval_topics}_{grps}_stats_avg_rscore_r100_new_gini.csv'
        if os.path.exists(res_csv):
            os.remove(res_csv)
            print(f'old {res_csv} removed')
                        
        print(f'saving {res_csv}')
        res_df.to_csv(res_csv, index=False)
        os.system(f'cp -r {res_csv} {prg_dir}/')
        print(f'copied into {prg_dir}/')



import sys
# cluster = int(sys.argv[1])
run_model = sys.argv[1:]
def cal_groups():
    # dev_topics['cluster'] = 0
    # grouped = dev_topics.groupby('cluster')
    # group_exec(run_model, [0], 1,'dev_all', grouped, dev_qrels, filter=False)

    # dl1920_topics['cluster'] = 0
    # dl1920_all = dl1920_topics.groupby('cluster')
    # group_exec(nc,'dl1920_all', dl1920_all, dl1920_qrels, filter=False)

    num_clusters = [500, 1000, 2000, 5000, 10000]
    for nc in num_clusters:
        print(f'For {nc} groups')
        csv = f'{prg_dir}/clustered_dev_queries_by_{nc}.csv'
        topics_sampled = pd.read_csv(csv, index_col=0).reset_index()
        grouped = topics_sampled.groupby('cluster')
        thres = [0, 30, 60, 90]
        group_exec(run_model, thres, nc, 'dev_topics_grps', grouped, dev_qrels, filter=True)

def convert_dev_all():
    res_df = pd.DataFrame()
    eval_topics = 'dev_all'
    group_nums = [1]
    for modelname in ['bm25', 'rtr_splade', 'tctcolbert', 'bm25_tctcolbert', 'bm25_monot5']:
        for grps in group_nums:
            csv = f'{data_dir}/allresults/result_{modelname}_{eval_topics}_{grps}_stats_avg_rscore_r100_new_gini.csv'
            print(f'loading {csv}')
            df = pd.read_csv(csv, index_col=0).reset_index()
            res_df = pd.concat([res_df, df], ignore_index=True)

    csv = f'{data_dir}/allresults/result_all_models_all_groups_avg_rscore_r100_new_gini.csv'
    res_df.to_csv(csv, index=False)
    os.system(f'cp -r {csv} {prg_dir}/')
    print(f'copied into {prg_dir}/')

def convert2dec4():
    res_df = pd.DataFrame()
    eval_topics = 'dev_topics_grps'
    group_nums = [500, 1000, 2000, 5000, 10000]
    for modelname in ['bm25', 'rtr_splade', 'tctcolbert', 'bm25_tctcolbert', 'bm25_monot5']:
        for grps in group_nums:
            csv = f'{data_dir}/allresults/result_{modelname}_{eval_topics}_{grps}_stats_avg_rscore_r100_new_gini.csv'
            print(f'loading {csv}')
            df = pd.read_csv(csv, index_col=0).reset_index()
            res_df = pd.concat([res_df, df], ignore_index=True)

    csv = f'{data_dir}/allresults/result_all_models_all_groups_avg_rscore_r100_new_gini.csv'
    res_df.to_csv(csv, index=False)
    os.system(f'cp -r {csv} {prg_dir}/')
    print(f'copied into {prg_dir}/')

cal_groups()
# convert2dec4()
# convert_dev_all()