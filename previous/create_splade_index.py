import warnings

from scipy.optimize import broyden1

warnings.filterwarnings('ignore')

import pyterrier as pt
if not pt.started():
    pt.init()

from ir_measures import AP, nDCG, P, R, RR, MRR
from DocumentFilter import *
from Lz4PickleCache import *

import pandas as pd
from pyterrier_caching import ScorerCache
import statsmodels.stats.weightstats

import os
import sys
import re
import numpy as np
import glob
import shutil
import json

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/datasets/cxj/retrievability-bias'

ranker = "splade"
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

msmarco_passage_index = f'{nfs_dir}/msmarco-passage-nostemmer-nostopwords-index'
indexed = pt.IndexRef.of(msmarco_passage_index)
index_ft = pt.IndexFactory.of(indexed).getCollectionStatistics().toString()
print("Index summary before pruning ---------------------")
print(index_ft)
total_corpus_size = int(re.findall(r"\d+\.?\d*", index_ft)[0])

import pyt_splade
splade = pyt_splade.Splade()

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

import json

def save(dict, file):
    with open(file,'w') as f:
        json.dump(dict, f)

def load(file):
    with open(file,'r') as f:
        ms_docids = json.load(f)
    return ms_docids

def calc_gini(modelname,df):
    scoredF = f'./results/{modelname}_docids_100.json'
    if not os.path.exists(scoredF):
        # csv = f'/nfs/datasets/cxj/retrievability-bias/results_{modelname}_100.csv'
        # df = pd.read_csv(csv)
        init_msmarco_dict = './results/ms_docids.json'
        docids_score = load(init_msmarco_dict)
        for qid in pt.tqdm(topics['qid']):
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
    # calc_stats(scores)
    gini_value = Gini(scores)
    return gini_value

#==============analysis===================
eval_metrics = eval_metrics=[nDCG@10, RR@10, 'map']
output_columns = ['name','nDCG@10', 'nDCG@10 +', 'nDCG@10 -', 'nDCG@10 p-value','RR@10', 'RR@10 +','RR@10 -', 'RR@10 p-value','map', 'map +', 'map -','map p-value']

import splade_retrieve as spladeretr
nopruned_index = f"{nfs_dir}/msmarco-passage-splade-nostemmer-nostopwords-index"
br1 = splade.query_encoder(batch_size=1000) >> pt.terrier.Retriever(nopruned_index, wmodel='Tf', verbose=True)

def create_index(inx, threshold, modelname, index_path, expt_path, cache_file, qual_signal=None):
    index_file = f"{nfs_dir}/splade/{modelname}-{ranker}-nostemmer-nostopwords-index-{str(threshold)}"

    print(f"indexing into {index_file}")
    pipe = DocumentFilter(qual_signal=qual_signal, threshold=threshold) >> splade.doc_encoder() >> pt.IterDictIndexer(index_file, stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, pretokenised=True, verbose=True)
    pipe.index(pt.tqdm(cache_file.get_corpus_iter()))

    print('indexing done')

fn = lambda X,Y: (0, statsmodels.stats.weightstats.ttost_paired(X, Y, -0.05, 1)[0])
def evaluate_experiment(inx, threshold, modelname, index_path, expt_path, cache_file, qual_signal=None):
    global br1
    global result_df_splade

    index_file = f"{index_path}-{modelname}-splade-nostemmer-nostopwords-index-{str(threshold)}"
    # if os.path.exists(index_file): # remove it in this experiment.
    #     shutil.rmtree(index_file)

    if inx == 0:
        indexref = pt.IndexRef.of(nopruned_index)
    elif os.path.exists(index_file):
        indexref = pt.IndexRef.of(index_file)
    else:
        print(f"indexing into {index_file}")
        pipe = DocumentFilter(qual_signal=qual_signal, threshold=threshold) >> splade.doc_encoder() >> pt.IterDictIndexer(index_file, stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, pretokenised=True, verbose=True)
        indexref = pipe.index(pt.tqdm(cache_file.get_corpus_iter()))

    # index_ft = pt.IndexFactory.of(indexref)
    # # Statistics
    # pruned_summary = index_ft.getCollectionStatistics().toString()
    # print(pruned_summary)
    # left_corpus_size = int(re.findall(r"\d+\.?\d*", pruned_summary)[0])
    # pruned_percentage = round(100 - left_corpus_size / total_corpus_size * 100)
    # print("pruned_percentage: ", pruned_percentage, "%")
    #
    # if inx == 0:
    #     br2 = br1
    # else:
    #     cache_dir = f"/nfs/datasets/cxj/retrievability-bias/contriever/cache/splade_cache_{str(threshold)}"
    #     br2 = spladeretr.get_cached_splade(indexref, cache_dir)
    #
    # br1 = br1 % retrieve_num
    # br2 = br2 % retrieve_num
    # print('start calc gini')
    # df2 = br2.transform(topics)
    # gini_value = calc_gini('splade', df2)
    # print(f'gini_value:{gini_value}')
    #
    # # Experiment dir
    # expt_file = f"{expt_path}-{modelname}-expt-splade-{str(threshold)}"
    # print(f"experiment into {expt_file}")
    # if os.path.exists(expt_file):
    #     shutil.rmtree(expt_file)
    # else:
    #     os.makedirs(expt_file)
    #
    # res = pt.Experiment(
    #     [br1, br2],
    #     topics,
    #     qrels,
    #     names=[f'NoPruned', f'{modelname.upper()}_Pruned'],
    #     eval_metrics=eval_metrics,
    #     save_dir=expt_file,
    #     baseline=0,
    #     # correction='bonferroni',
    #     test=fn
    # )
    #
    # res = res[output_columns]
    # res.insert(1, 'Percentage', pruned_percentage)
    # res.insert(2, 'Threshold', threshold)
    # res.insert(3, 'gini_value', gini_value)
    # res = res.round(4)
    # print(res)
    # print('starting to save splade csv')
    # res.to_csv(f'{nfs_dir}/contrast/{modelname}_dev_splade_{str(threshold)}.csv')
    # result_df_splade = pd.concat([result_df_splade, res], ignore_index=True)
    #
    # os.popen(f'cp -r {expt_file} {nfs_dir}/contrast/')
    # print(f'{expt_file} copied.')

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

result_df_splade = pd.DataFrame()
modelname = 't5-base-msmarco-epoch-5'
index_path = f'{root_dir}/msmarco_dev-{ranker}'
expt_path = f'{root_dir}/msmarco_dev-{ranker}'

root_file_name = f'{root_dir}/{modelname}.lz4'
if not os.path.exists(root_file_name):
    os.system(f'cp /nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4 {root_dir}/')
cache_file = Lz4PickleCache(f'{root_dir}/{modelname}.lz4')

pert_file = f'{nfs_dir}/contrast/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30 / 5, 60 / 5, 90 / 5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

for i, threshold in enumerate(percent):
    if i == 0:
        continue

    create_index(i, threshold, modelname, index_path, expt_path, cache_file, qual_signal="prob")
    # evaluate_experiment(i, threshold, modelname, index_path, expt_path, cache_file, qual_signal="prob")


# file_splade = f'{root_dir}/{modelname}_dev_{ranker}.csv'
# print(f'saving {file_splade}')
# result_df_splade.to_csv(file_splade, index=False)
# os.popen(f'cp -r {file_splade} {nfs_dir}/contrast/')
# print('copied.')

