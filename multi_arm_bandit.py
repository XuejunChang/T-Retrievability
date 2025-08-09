import pandas as pd, time, os
import numpy as np
from collections import defaultdict
import evaluate, config, convert
from sklearn.metrics import ndcg_score, average_precision_score
import argparse

retr_num = 100
freq_num = 100 # threshold of retrieved number of documents
cut_off = 20

eps = 0.1
# iter_num = 2094870 # The number of trials
iter_num = 100000 # The number of trials

def mab(modelname):
    res0 = f'{config.data_dir}/{modelname}_msmarco-passage_dev_{retr_num}.res'
    print(f'processing {res0}')
    columns = ['qid', 'Q0', 'docid', 'rank', 'score', 'run']
    df = convert.convert_res2docdf(res0, columns=columns)
    df0_cut20 = df.groupby("qid", group_keys=False).head(cut_off).reset_index()
    res0_cut20 = f'{config.data_dir}/{modelname}_msmarco-passage_dev_{retr_num}_cut{cut_off}.res'
    convert.convert_docdf2_trec(df0_cut20,res0_cut20,'df0_cut20')

    qrels_res = f'{config.data_dir}/qrels_dev.res'
    metrics = evaluate.evaluate_metrics(qrels_res, res0_cut20)
    baseline_ndcg, baseline_map = metrics["ndcg_cut_10"], metrics["map"]

    # construct candidate docs
    doc_freq = df['docid'].value_counts()
    candidates = doc_freq[doc_freq >= freq_num].index.tolist()
    # initialize MAB state
    values = defaultdict(float)
    counts = defaultdict(int)
    print(f'bandit by {iter_num}')
    for i in range(iter_num):
        start = time.time()
        # Epsilon-greedy selection
        if np.random.rand() < eps:
            arm = np.random.choice(candidates)
        else:
            arm = max(candidates, key=lambda x: values[x])

        # try to remove this doc
        df_temp = df[df['docid'] != arm]
        df_temp = df_temp.groupby("qid", group_keys=False).head(cut_off).reset_index()

        df_temp_res = f'{config.data_dir}/{modelname}_msmarco-passage_dev_100_temp_iter{i}_cut{cut_off}.res'
        convert.convert_docdf2_trec(df_temp, df_temp_res, f'df_temp_iter{i}_cut{cut_off}')
        print(f'calc metrics for {df_temp_res}')
        metrics = evaluate.evaluate_metrics(qrels_res, df_temp_res)
        os.remove(df_temp_res)
        print(f'removed {df_temp_res}')

        # current metrics when removed
        ndcg, map = metrics["ndcg_cut_10"], metrics["map"]
        ndcg_drop = abs(baseline_ndcg - ndcg) / baseline_ndcg
        map_drop = abs(baseline_map - map) / baseline_map
        metric_drop = (ndcg_drop + map_drop) / 2
        reward = 1 - metric_drop  # smaller drop get higher reward

        # updating statistic of MAB
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]

        # max 1% degradation, remove arm if drop is too high
        if reward < 0.99:
            candidates.remove(arm)

        end = time.time()
        print(f'time: {(end - start):.2f} seconds')

    # remove all selected arms from df
    mab_df = df[~df['docid'].isin(counts.keys())]
    mab_df_res = f'{config.data_dir}/{modelname}_msmarco-passage_dev_{retr_num}_mab_iteration_{iter_num}.res'
    run_name = os.path.splitext(mab_df_res)[0].split('/')[-1]
    convert.convert_docdf2_trec(mab_df, mab_df_res, run_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Accept multiple methods and models")
    parser.add_argument('--models', nargs='+', help='Which models to be ran')
    args = parser.parse_args()
    models = args.models

    for modelname in models:
        mab(modelname)

