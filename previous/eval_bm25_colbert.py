import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from Lz4PickleCache37 import *
import pandas as pd
import os
import numpy as np

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

ranker = "colbert"
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

import json
import statistics
def save(dict, file):
    with open(file,'w') as f:
        json.dump(dict, f)

def load(file):
    with open(file,'r') as f:
        ms_docids = json.load(f)
    return ms_docids

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

import tqdm
def calc_stats(modelname,df, threshold, topics):
    scoredF = f'./results/{modelname}_docids_100_threshold_{threshold}.json'
    if not os.path.exists(scoredF):
        init_msmarco_dict = './results/ms_docids.json'
        docids_score = load(init_msmarco_dict)
        for qid in tqdm.tqdm(topics['qid']):
            D = df[df['qid'] == np.int64(qid)]
            for dno in D['docno']:
                rank = D[D['docno'] == dno]['rank'].values[0]
                score = 100 / np.log(rank + 2)  # plus 2 because the ranks start from zero
                docids_score[str(dno)] += score
        save(docids_score, scoredF)

    docids_score = load(scoredF)
    scores_df = pd.DataFrame.from_dict(docids_score, orient="index", columns=["score"])
    scores_df = scores_df[scores_df["score"] > 0]
    scores = scores_df['score'].to_list()

    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    gini_value = Gini(scores)
    return mean, std_dev, gini_value

import ir_measures
from ir_measures import *
def calc_irmetrics(df):
    df = df.rename(columns={'qid':'query_id','docid':'doc_id'})
    df[['query_id','doc_id']] = df[['query_id','doc_id']].astype(str)
    this_dict = ir_measures.calc_aggregate([RR, nDCG@10], qrels, df)
    return this_dict['nDCG@10'], this_dict['RR']


from pyterrier_colbert.ranking import ColBERTFactory
CHECKPOINT = "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
def evaluate_experiment(inx, threshold, modelname,qstart,qend):
    index_path = f"{nfs_dir}/{ranker}/{modelname}-{ranker}-index-threshold-{threshold}"
    if not os.path.exists(index_path):
        print(f"Index file {index_path} does not exist")
        return

    index_dir = '/'.join(index_path.split('/')[:-1])
    index_name = index_path.split('/')[-1]
    pytcolbert = ColBERTFactory(CHECKPOINT, f"{index_dir}", index_name, faiss_partitions=100)

    csv = f'{nfs_dir}/{ranker}/df_bm25_{ranker}_{inx * 30}_{qstart}.csv'
    if os.path.exists(csv):
        print(f"Index file {csv} already exists")
        return

    bm25_csv = f'{nfs_dir}/bm25/df_bm25_{inx * 30}.csv'
    if os.path.exists(bm25_csv):
        bm25_df = pd.read_csv(bm25_csv,index_col=0).reset_index()
        bm25_df[['qid', 'docno']] = bm25_df[['qid', 'docno']].astype(str)
        retriever = pt.text.get_text(dataset, 'text') >> pytcolbert.text_scorer(verbose=True)

        print(f'start tramsforming bm25_df >> {ranker} threshold {threshold}')
        qstart, qend = qstart * retrieve_num, qend * retrieve_num
        print(f'start: {qstart}, end: {qend}')
        if qend < 0:
            qend = bm25_df.shape[0]
        df = retriever.transform(bm25_df[qstart:qend])
        print(f'df {ranker} columns {df.columns.tolist()}')
        # cols = ['qid', 'docid', 'docno', 'score', 'rank']
        # print(f'to opt in columns {cols}')
        # df = df[cols]
        df.to_csv(csv, index=False)
        print(f'saved bm25>>{ranker} with shape {df.shape} into {csv}')

    # else:
    #     # cache_bm25_dir = f"{nfs_dir}/bm25/bm25_cache_{str(threshold)}"
    #     index_path = f'{nfs_dir}/bm25/t5-base-msmarco-epoch-5-nostemmer-nostopwords-index-{threshold}'
    #     # cached_bm25 = utils.get_cached_bm25(None, cache_bm25_dir)
    #     retriever = pt.BatchRetrieve(index_path, wmodel='BM25', verbose=True)
    #     retriever = retriever  % retrieve_num >> pt.text.get_text(dataset, 'text') >> pytcolbert.text_scorer()
    #
    #     print(f'start tramsforming bm25 >> {ranker}')
    #     if qend < 0:
    #         qend = topics.shape[0]
    #     df = retriever.transform(topics[qstart:qend])
    #     df.to_csv(csv, index=False)
    #     print(f'saved bm25>>{ranker} with shape {df.shape} into {csv}')



    # print('start calc bm25 >> {ranker} metrics')
    # mean, std_dev, gini_value = calc_stats(f'bm25_{ranker}', df, threshold, topics)
    # nDCG, RR = calc_irmetrics(df)
    # print(f'bm25 >> {ranker} mean: {mean}, std_dev: {std_dev}, gini_value:{gini_value}')
    # print(f'bm25 >> {ranker} nDCG: {nDCG}, RR: {RR}')
    #
    # result_df = pd.DataFrame([[mean, std_dev, gini_value, nDCG, RR]],columns=['mean', 'std_dev', 'gini_value', 'nDCG', 'RR'])
    # csv = f'{nfs_dir}/{ranker}/result_bm25_{ranker}_{inx*30}.csv'
    # result_df.to_csv(csv,index=False)
    # print(f'save to {csv}')

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

modelname = 't5-base-msmarco-epoch-5'
cache_file = Lz4PickleCache(f'/nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4')

pert_file = f'{nfs_dir}/bm25/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30/5, 60/5, 90/5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

import sys
qstart,qend = int(sys.argv[1]), int(sys.argv[2])
for i, threshold in enumerate(percent):
    if i < 2:
        continue
    evaluate_experiment(i, threshold, modelname, qstart,qend)
