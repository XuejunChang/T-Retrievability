import warnings

warnings.filterwarnings('ignore')

import pyterrier as pt
if not pt.started():
    pt.init()

from ir_measures import nDCG, RR
from DocumentFilter import *
from Lz4PickleCache import *

import pandas as pd
from pyterrier_caching import ScorerCache
import statsmodels.stats.weightstats

import os
import numpy as np
import shutil

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

ranker = "contriever"
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

# msmarco_passage_index = f'{nfs_dir}/msmarco-passage-nostemmer-nostopwords-index'
# indexed = pt.IndexRef.of(msmarco_passage_index)
# index_ft = pt.IndexFactory.of(indexed).getCollectionStatistics().toString()
# print("Index summary before pruning ---------------------")
# print(index_ft)
# total_corpus_size = int(re.findall(r"\d+\.?\d*", index_ft)[0])
total_corpus_size = 8841823
#==============analysis===================
eval_metrics = eval_metrics=[nDCG@10, RR@10, 'map']
output_columns = ['name','nDCG@10', 'nDCG@10 +', 'nDCG@10 -', 'nDCG@10 p-value','RR@10', 'RR@10 +','RR@10 -', 'RR@10 p-value','map', 'map +', 'map -','map p-value']

from contriever_model import Contriever
import pyterrier_dr
from pyterrier_caching import RetrieverCache
import contriever_retrieve_e2e as contre2e

model_name = "facebook/contriever-msmarco"
tokenizer = "facebook/contriever-msmarco"
model = Contriever(model_name, batch_size=32)

# nopruned_index = f'{nfs_dir}/{ranker}/pt_contriever_index/contriever.flex'
nopruned_index = None
# index = pyterrier_dr.FlexIndex(nopruned_index)
# br1 = model >> index.torch_retriever()
# import bm25_retrieve as bm25retr
# nopruned_bm25_path = f"/nfs/datasets/cxj/retrievability-bias/bm25/bm25_cache"
# nopruned_bm25_cache = RetrieverCache(nopruned_bm25_path)
# nopruned_pipeline = nopruned_bm25_cache % retrieve_num >> pt.text.get_text(dataset, 'text') >> model

def create_index(inx, threshold, modelname, index_path, expt_path, cache_file, qual_signal=None):
    index_file = f'{nfs_dir}/{ranker}/{modelname}-{ranker}-index-{str(threshold)}.flex'

    print(f"indexing into {index_file}")
    index = pyterrier_dr.FlexIndex(index_file)
    idx_pipeline = DocumentFilter(qual_signal=qual_signal, threshold=threshold) >> model >> index
    idx_pipeline.index(pt.tqdm(cache_file.get_corpus_iter()))

    print('index done')

fn = lambda X,Y: (0, statsmodels.stats.weightstats.ttost_paired(X, Y, -0.05, 1)[0])
def evaluate_experiment(inx, threshold, modelname, index_path, expt_path, cache_file, qual_signal=None):
    global br1
    global nopruned_pipeline
    global result_df_bm25_contriever
    global result_df_contriever

    index_file = f'{nfs_dir}/{ranker}/pt_contriever_index/contriever_{str(threshold)}.flex'

    if inx == 0:
        indexref = nopruned_index
    elif os.path.exists(index_file):
        indexref = index_file
    else:
        print(f"indexing into {index_file}")
        index = pyterrier_dr.FlexIndex(index_file)
        idx_pipeline = DocumentFilter(qual_signal=qual_signal, threshold=threshold) >> model >> index
        indexref = idx_pipeline.index(pt.tqdm(cache_file.get_corpus_iter()))

    pruned_percentage = i * 30
    print("pruned_percentage: ", pruned_percentage, "%")

    if inx == 0:
        br2 = br1
    else:
        cache_dir = f"/nfs/datasets/cxj/retrievability-bias/{ranker}/cache/contriever_cache_{str(threshold)}"
        br2 = contre2e.get_cached_contriever(indexref, cache_dir)

    br1 = br1 % retrieve_num
    br2 = br2 % retrieve_num
    print('start calc tramsform {ranker}}')
    df2 = br2.transform(topics)
    df2 = df2[['qid','docid','docno','score','rank']]  # don't need this column to calc retrievability score
    csvfile = f'{nfs_dir}/{ranker}/results_contriever_raw_{str(threshold)}.csv'
    print(f'starting to save {csvfile}')
    df2.to_csv(csvfile,index=False)

    print('start calc {ranker}} gini')
    gini_value = utils.calc_gini("contriever", df2, threshold, topics)
    print(f'{ranker} gini_value:{gini_value}')

    # Experiment dir
    expt_file = f"{expt_path}-{modelname}-expt-{ranker}-{str(threshold)}"
    print(f"experiment into {expt_file}")
    if os.path.exists(expt_file):
        shutil.rmtree(expt_file)
    else:
        os.makedirs(expt_file)

    res = pt.Experiment(
        [br1, br2],
        topics,
        qrels,
        names=[f'NoPruned', f'{modelname.upper()}_Pruned'],
        eval_metrics=eval_metrics,
        save_dir=expt_file,
        baseline=0,
        # correction='bonferroni',
        test=fn
    )

    res = res[output_columns]
    res.insert(1, 'Percentage', pruned_percentage)
    res.insert(2, 'Threshold', threshold)
    res.insert(3, 'gini_value', gini_value)
    res = res.round(4)
    print(res)
    csvfile = f'{nfs_dir}/{ranker}/{modelname}_expt_{ranker}_{str(threshold)}.csv'
    print('starting to save {csvfile}')
    res.to_csv(csvfile, index=False)
    result_df_contriever = pd.concat([result_df_contriever, res], ignore_index=True)

    os.popen(f'cp -r {expt_file} {nfs_dir}/{ranker}/')
    print(f'{expt_file} copied.')

    expt_file = f"{expt_path}-{modelname}-expt-BM25-{ranker}-{str(threshold)}"
    print(f"experiment into {expt_file}")
    if os.path.exists(expt_file):
        shutil.rmtree(expt_file)
    else:
        os.makedirs(expt_file)

    if inx == 0:
        pipeline2 = nopruned_pipeline
    else:
        cachedbm25 = RetrieverCache(f"/nfs/datasets/cxj/retrievability-bias/bm25/bm25_cache_{str(threshold)}")
        pipeline2 = cachedbm25 % retrieve_num >> pt.text.get_text(dataset, 'text') >> model

    print('start transform bm25>>{ranker}')
    df_pipe2 = pipeline2.transform(topics)
    df_pipe2 = df_pipe2[['qid', 'docid', 'docno', 'score', 'rank']]
    csvfile = f'{nfs_dir}/{ranker}/results_bm25_contriever_raw_{str(threshold)}.csv'
    print(f'start to save {csvfile}')
    df_pipe2.to_csv(csvfile,index=False)
    print(f'save {csvfile} done')
    print('start calc bm25>>{ranker} gini')
    gini_pipe2 = utils.calc_gini('bm25_contriever', df_pipe2, threshold, topics)
    print('bm25>>{ranker} gini: {gini_pipe2}')
    res = pt.Experiment(
        [nopruned_pipeline, pipeline2],
        topics,
        qrels,
        names=[f'BM25_Colbert', f'{modelname.upper()}_Pruned_BM25_Colbert'],
        eval_metrics=eval_metrics,
        save_dir=expt_file,
        baseline=0,
        # correction='bonferroni',
        test=fn
    )

    res = res[output_columns]
    res.insert(1, 'Percentage', pruned_percentage)
    res.insert(2, 'Threshold', threshold)
    res.insert(3, 'gini_value', gini_pipe2)
    res = res.round(4)
    print(res)
    csvfile = f'{nfs_dir}/{ranker}/{modelname}_expt_bm25_{ranker}_{str(threshold)}.csv'
    print(f'starting to save {csvfile}')
    res.to_csv(csvfile, index=False)
    result_df_bm25_contriever = pd.concat([result_df_bm25_contriever, res], ignore_index=True)

    os.popen(f'cp -r {expt_file} {nfs_dir}/{ranker}/')
    print(f'{expt_file} copied.')

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

result_df_contriever = pd.DataFrame()
result_df_bm25_contriever = pd.DataFrame()
modelname = 't5-base-msmarco-epoch-5'
index_path = f'{root_dir}/msmarco_dev-{ranker}'
expt_path = f'{root_dir}/msmarco_dev-{ranker}'

root_file_name = f'{root_dir}/{modelname}.lz4'
if not os.path.exists(root_file_name):
    os.system(f'cp /nfs/resources/cxj/retrievability-bias/supervisedT5/{modelname}.lz4 {root_dir}/')
cache_file = Lz4PickleCache(f'{root_dir}/{modelname}.lz4')

pert_file = f'{nfs_dir}/bm25/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30/5, 60/5, 90/5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

for i, threshold in enumerate(percent):
    if i>0:
        break
    # evaluate_experiment(i, threshold, modelname, index_path, expt_path, cache_file, qual_signal="prob")
    create_index(i, threshold, modelname, index_path, expt_path, cache_file, qual_signal="prob")


# csvfile = f'{root_dir}/{modelname}_dev_{ranker}.csv'
# print(f'saving {csvfile}')
# result_df_contriever.to_csv(csvfile, index=False)
# os.popen(f'cp -r {csvfile} {nfs_dir}/{ranker}/')
# print('copied.')
#
# csvfile = f'{root_dir}/{modelname}_dev_bm25_{ranker}.csv'
# print(f'saving {csvfile}')
# result_df_bm25_contriever.to_csv(csvfile, index=False)
# os.popen(f'cp -r {csvfile} {nfs_dir}/{ranker}/')
# print('copied.')



