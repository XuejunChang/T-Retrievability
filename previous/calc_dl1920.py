import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

import pyterrier as pt
if not pt.java.started():
    pt.java.init()
import statistics

import os
os.environ["PIP_ROOT_USER_ACTION"] = "ignore"

dataset = pt.get_dataset(f'irds:msmarco-passage')
# df_dataset = pd.DataFrame(dataset.get_corpus_iter(verbose=True))
eval_dev = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dev.get_topics()
# qrels = eval_dev.get_qrels()

dl19 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019')
dl19_topics = dl19.get_topics()
dl20 = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')
dl20_topics = dl20.get_topics()

dl1920 = pd.concat([dl19_topics, dl20_topics], ignore_index=True)

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


# def calc_stats(modelname,df,scoredF, topics):
#     if not os.path.exists(scoredF):
#         init_msmarco_dict = './results/ms_docids.json'
#         docids_score = load(init_msmarco_dict)
#         for qid in pt.tqdm(topics['qid']):
#             D = df[df['qid'] == np.int64(qid)]
#             for dno in D['docno']:
#                 rank = D[D['docno'] == dno]['rank'].values[0]
#                 score = 100 / np.log(rank + 2)  # plus 2 because the ranks start from zero
#                 docids_score[str(dno)] += score
#         save(docids_score, scoredF)

#     docids_score = load(scoredF)
#     scores_df = pd.DataFrame.from_dict(docids_score, orient="index", columns=["score"])
#     scores_df = scores_df[scores_df["score"] > 0]
#     scores = scores_df['score'].to_list()

#     mean = statistics.mean(scores)
#     std_dev = statistics.stdev(scores)
#     gini_value = Gini(scores)
#     return mean, std_dev, gini_value


def calc_stats_v2(modelname,df,scoredF, topics):
    if not os.path.exists(scoredF):
        qids_to_keep = topics['qid'].to_list()
        mask = np.logical_or.reduce([df["qid"] == val for val in qids_to_keep])
        df_filtered = df[mask]
        grouped = df_filtered.groupby("docno")[['r_score']].sum().reset_index()
        grouped.to_csv(scoredF,index=False)
    else:
        grouped = pd.read_csv(scoredF, index_col=0).reset_index()

    scores = grouped['r_score'].to_list()

    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    gini_value = Gini(scores)
    return mean, std_dev, gini_value


root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

# not clustered
all_topics = ['dl1920']
pt.tqdm.pandas()

for eval_topics in all_topics:
    # for modelname in ['bm25', 'bm25_monot5', 'splade', 'colbert', 'bm25_colbert']:
    for modelname in ['bm25', 'bm25_monot5', 'splade']:
        res_gini = []
        for threshold in [0, 30, 60, 90]:
            """
            Calc retrievability score for each doc
            """
            print(f'start {modelname} ----> {eval_topics} ----> threshold {threshold}')
            rscore_csv = f'/nfs/resources/cxj/retrievability-bias/{modelname}/df_{modelname}_rscore_{eval_topics}_{threshold}.csv'
            if os.path.exists(rscore_csv):
                df = pd.read_csv(rscore_csv, index_col=0).reset_index()
            else:
                csv = f'/nfs/resources/cxj/retrievability-bias/{modelname}/df_{modelname}_{eval_topics}_{threshold}.csv'
                df = pd.read_csv(csv, index_col=0).reset_index()
                df['r_score'] = df['rank'].progress_apply(lambda x: 100 / np.log(x + 2))
                print(f'saving {rscore_csv}')
                df.to_csv(rscore_csv, index=False)
                print(f'done')

            """
            Calc stats for each group 
            """
            res = []
            print(f'Calc stats {modelname} ----> {eval_topics} ----> threshold {threshold}')
            scoredF = f'{nfs_dir}/{modelname}/groups/{modelname}_{eval_topics}_T{threshold}.csv'
            mean, std, gini = calc_stats_v2(modelname, df, scoredF, dl1920)
            group_res = [modelname, threshold, mean, std, gini]
            print(group_res)
            res.append(group_res)

        #     """
        #     merge for each threshold
        #     """
        #     print(f'merge for each threshold for {modelname} ----> {eval_topics} ----> threshold {threshold}')
        #     df_threshold = pd.DataFrame(res, columns=['modelname', 'threshold', 'mean', 'std', 'gini'])
        #     res_csv = f'{nfs_dir}/{modelname}/groups/result_{eval_topics}_T{threshold}_allgroups.csv'
        #     print(f'saving {res_csv}')
        #     df_threshold.to_csv(res_csv, index=False)
        #     print('done')
        #
        #     """
        #     Calc ginis for each threshold
        #     """
        #     print(f'Calc ginis for each threshold for {modelname} ----> {eval_topics} ----> threshold {threshold}')
        #     ginis = df_threshold['gini']
        #     min_gini, mean_gini, max_gini = ginis.min(), ginis.mean(), ginis.max()
        #     res_gini.append([modelname, threshold, min_gini, mean_gini, max_gini])
        #
        # """
        # Merge into one file
        # """
        # print(f'Merge into one file for {modelname} ----> {eval_topics}')
        # res_df = pd.DataFrame(res_gini, columns=['modelname', 'threshold', 'min_gini', 'mean_gini', 'max_gini'])
        # res_csv = f'{nfs_dir}/allresults/result_{modelname}_{eval_topics}_stats.csv'
        # print(f'saving {res_csv}')
        # res_df.to_csv(res_csv, index=False)
        # os.system(f'cp -r {res_csv} {root_dir}/results/')
        # print(f'copied {res_csv}')
