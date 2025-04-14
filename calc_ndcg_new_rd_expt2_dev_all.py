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

def calc_eff(df,qrels):
    df['query_id'] = df['qid'].astype(str)
    df['doc_id'] = df['docno'].astype(str)
    m = ir_measures.calc_aggregate([nDCG @ 10, RR], qrels, df)

    return m[nDCG @ 10], m[RR]


def calc_r_gini_stats(df, topics, scoredF, filter=False):
    # # if os.path.exists(scoredF):
    # #     os.remove(scoredF)
    # #     print(f'old scoredF: {scoredF} removed')
    #
    # if filter:
    #     print('filtering')
    #     qids_to_keep = topics['qid'].to_list()
    #     mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
    #     df = df[mask]
    #
    # print('groupby docno ...')
    # grouped = df.groupby("docno")[['r_score']].sum().reset_index()

    if not os.path.exists(scoredF):
        if filter:
            print('filtering')
            qids_to_keep = topics['qid'].to_list()
            mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
            df = df[mask]

        print('groupby docno ...')
        grouped = df.groupby("docno")[['r_score']].sum().reset_index()

        grouped['r_score'] = grouped['r_score']/len(qids_to_keep)
        print(f'save to {scoredF}')
        grouped.to_csv(scoredF, index=False)
    else:
        grouped = pd.read_csv(scoredF, index_col=0).reset_index()
        print(f'loaded from {scoredF}')

    print('start statistics')
    scores = grouped['r_score'].to_list()
    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    gini_value = Gini(scores)

    return mean, std_dev, gini_value

nfs_dir = f'/nfs/resources/cxj/retrievability-bias'
# nfs_dir = f'/nfs/datasets/cxj/retrievability-bias-from_resources_ok'
root_dir = '/root/retrievability-bias'
os.makedirs(f'{root_dir}/allresults', exist_ok=True)
pri_dir = '/nfs/primary/retrievability-bias/results/new_clustered'
def group_exec(run_model, grps,eval_topics, grouped, qrels, filter=False):
    # for modelname in ['bm25', 'bm25_monot5', 'rtr_splade', 'tctcolbert','bm25_tctcolbert']:
    for modelname in [run_model]:
        os.makedirs(f'{root_dir}/{modelname}/groups/new_clustered/', exist_ok=True)
    # for modelname in ['tctcolbert', 'bm25_tctcolbert']:
        this_model_res = []
        for threshold in [0, 30, 60, 90]:
            """
            Calc retrievability score for each doc
            """
            print(f'start  {modelname} ----> {eval_topics} --number of groups {grps} ----> threshold {threshold}')
            # origin_topics = eval_topics.split('_')[0]
            origin_topics = 'dev_topics'
            # origin_topics = 'dev'
            rscore_csv = f'{nfs_dir}/{modelname}/df_{modelname}_{origin_topics}_rscore_{threshold}_v2.csv'
            # rscore_csv = f'{nfs_dir}/{modelname}/df_{modelname}_{origin_topics}_rscore_{threshold}_firstexpt_v2.csv'
            if os.path.exists(rscore_csv):
                print(f'loading {rscore_csv}')
                df = pd.read_csv(rscore_csv, index_col=0).reset_index()
            else:
                csv = f'{nfs_dir}/{modelname}/df_{modelname}_{origin_topics}_{threshold}.csv'
                df = pd.read_csv(csv, index_col=0).reset_index()
                # df['r_score'] = df['rank'].progress_apply(lambda x: 100 / np.log(x + 2))
                df['r_score'] = df['rank'].progress_apply(lambda x: 1 / np.log(x + 2))
                print(f'saving {rscore_csv}')
                df.to_csv(rscore_csv, index=False)
                print(f'done')

            """
            Calc stats for each group 
            """
            res = []
            for group_id, queries_df in grouped:
                print(f'Calc {modelname} ----> {eval_topics} --> number of groups {grps} ---> group_id={group_id} ----> threshold {threshold}')
                scoredF = f'{root_dir}/{modelname}/groups/{modelname}_{eval_topics}_{grps}_G{group_id}_T{threshold}_v2.csv'
                # scoredF = f'{root_dir}/{modelname}/groups/{modelname}_{eval_topics}_{grps}_G{group_id}_T{threshold}_firstexpt_v2.csv'
                mean, std_dev, gini_value = calc_r_gini_stats(df, queries_df, scoredF, filter=filter)
                nDCG10, rr = calc_eff(df, qrels) if not filter else (None, None)
                group_res = [modelname, threshold, grps, group_id, mean, std_dev, gini_value, nDCG10, rr]
                res.append(group_res)


            print(f'merge into a dataframe for {modelname} ----> {eval_topics} ---> nubmer of groups:{grps} ---> group_id={group_id} ----> threshold {threshold}')
            df_threshold = pd.DataFrame(res, columns=['modelname', 'threshold','grps', 'group_id', 'mean', 'std', 'gini',
                                                      'nDCG@10', 'RR'])
            res_csv = f'{root_dir}/{modelname}/groups/new_clustered/result_{modelname}_{eval_topics}_{grps}_{group_id}_T{threshold}_v2.csv'
            # res_csv = f'{root_dir}/{modelname}/groups/new_clustered/result_{modelname}_{eval_topics}_{grps}_{group_id}_T{threshold}_firstexpt_v2.csv'
            print(f'saving {res_csv}')
            df_threshold.to_csv(res_csv, index=False)
            print('done')

            """
            statistics for this threshold 
            """
            print(f'Calc ginis for each threshold for {modelname} ----> {eval_topics} ------> nubmer of groups:{grps} ----> threshold {threshold}')
            ginis = df_threshold['gini']
            min_gini, mean_gini, max_gini = ginis.min(), ginis.mean(), ginis.max()
            nDCG10 = df_threshold['nDCG@10'].mean()
            rr = df_threshold['RR'].mean()
            this_model_res.append([modelname, threshold, grps, min_gini, mean_gini, max_gini, nDCG10, rr])

        """
        Merge all thresholds for this model.
        """
        print(f'Merge into one file for {modelname} ----> {eval_topics} ------> nubmer of groups:{grps}')
        res_df = pd.DataFrame(this_model_res,
                              columns=['modelname', 'threshold', 'grps', 'min_gini', 'mean_gini', 'max_gini', 'nDCG@10', 'RR'])
        res_df = res_df.round(4)
        res_df = res_df.sort_values(by='grps')

        res_csv = f'{root_dir}/allresults/result_{modelname}_{eval_topics}_{grps}_stats_v2.csv'
        # res_csv = f'{root_dir}/allresults/result_{modelname}_{eval_topics}_{grps}_stats_firstexpt_v2.csv'
        print(f'saving {res_csv}')
        res_df.to_csv(res_csv, index=False)
        os.system(f'cp -r {res_csv} {pri_dir}/')
        print(f'copied into {pri_dir}/')
import sys
# cluster = int(sys.argv[1])
run_model = sys.argv[1]
def cal_groups():
    num_clusters = [500, 1000, 2000, 5000, 10000]
    # num_clusters = [cluster]
    # num_clusters = [1000]
    # num_clusters = [2000]
    # num_clusters = [5000]
    # num_clusters = [10000]
    # group_size = [200,100,50,20,10]
    # groups = [2000]
    # group_size = [50]

    for nc in num_clusters:
        # all_topics = ['dl1920_all', 'dev_all', 'dev_2000_grps']

        # dl1920_topics['cluster'] = 0
        # dl1920_all = dl1920_topics.groupby('cluster')
        # group_exec(run_model, nc,'dl1920_all', dl1920_all, dl1920_qrels, filter=False)

        dev_topics['cluster'] = 0
        dev_all = dev_topics.groupby('cluster')
        group_exec(run_model, nc,'dev_all', dev_all, dev_qrels, filter=False)

        # print(f'For {nc} groups')
        # csv = f'{pri_dir}/clustered_dev_queries_by_{nc}.csv'
        # topics_sampled = pd.read_csv(csv, index_col=0).reset_index()
        # dev_topics_grps = topics_sampled.groupby('cluster')
        # group_exec(run_model, nc, 'dev_topics_grps', dev_topics_grps, dev_qrels, filter=True)

def convert2dec4():
    res_df = pd.DataFrame()
    eval_topics = 'dev_topics_grps'
    group_nums = [500, 1000, 2000, 5000, 10000]
    for modelname in ['bm25', 'bm25_monot5', 'splade', 'tctcolbert', 'bm25_tctcolbert']:
        for grps in group_nums:
            csv = f'{pri_dir}/result_{modelname}_{eval_topics}_{grps}_stats.csv'
            print(f'start {csv}')
            df = pd.read_csv(csv, index_col=0).reset_index()
            res_df = pd.concat([res_df, df], ignore_index=True)

    csv = f'{pri_dir}/result_all_models_all_groups.csv'
    res_df.to_csv(csv, index=False)

cal_groups()
# convert2dec4()