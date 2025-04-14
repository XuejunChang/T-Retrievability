import numpy as np
import json
import pandas as pd
import tqdm
import os
import statistics
import ir_measures
from ir_measures import *
import ir_datasets
eval = ir_datasets.load("msmarco-passage/dev")
# topics = pd.DataFrame(eval.queries_iter())
qrels = pd.DataFrame(eval.qrels_iter())

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

# def calc_stats(df, threshold, topics):
#     init_msmarco_dict = './results/ms_docids.json'
#     docids_score = load(init_msmarco_dict)
#     print('start calculating stats')
#     for qid in tqdm.tqdm(topics['qid']):
#         D = df[df['qid'] == np.int64(qid)]
#         for dno in D['docno']:
#             rank = D[D['docno'] == dno]['rank'].values[0]
#             score = 100 / np.log(rank + 2)  # plus 2 because the ranks start from zero
#             docids_score[str(dno)] += score
#     scores_df = pd.DataFrame.from_dict(docids_score, orient="index", columns=["score"])
#     scores_df = scores_df[scores_df["score"] > 0]
#     scores = scores_df['score'].to_list()
#
#     mean = statistics.mean(scores)
#     std_dev = statistics.stdev(scores)
#     gini_value = Gini(scores)
#     return mean, std_dev, gini_value

def calc_irmetrics(df):
    df = df.rename(columns={'qid':'query_id','docid':'doc_id'})
    df[['query_id','doc_id']] = df[['query_id','doc_id']].astype(str)
    this_dict = ir_measures.calc_aggregate([RR, nDCG@10], qrels, df)
    return this_dict['nDCG@10'], this_dict['RR']


import pyterrier as pt
from pyterrier_caching import RetrieverCache
def get_cached_bm25(index_file, cache_dir):
    print(f'checking cache dir {cache_dir}')
    if not os.path.exists(cache_dir):
        print('start caching bm25 retrieval')
        retriever = pt.terrier.Retriever(index_file, wmodel='BM25', verbose=True)
        cached_bm25 = RetrieverCache(cache_dir, retriever)
    else:
        cached_bm25 = RetrieverCache(cache_dir)
    return cached_bm25

def get_cached_splade(index_path, cache_dir,model):
    if not os.path.exists(cache_dir):
        print('start splade retrieval caching')
        retriever = model.query_encoder() >> pt.terrier.Retriever(index_path, wmodel='Tf', verbose=True)
        cached_splade = RetrieverCache(cache_dir, retriever)
    else:
        cached_splade = RetrieverCache(cache_dir)

    return cached_splade


import pyterrier_dr
def get_cached_contriever(index_path, cache_dir, model):
    if not os.path.exists(cache_dir):
        print('start contriever retrieval caching')
        index = pyterrier_dr.FlexIndex(index_path)
        retriever = model >> index.torch_retriever()
        cached_contriever = RetrieverCache(cache_dir, retriever)
    else:
        cached_contriever = RetrieverCache(cache_dir)

    return cached_contriever