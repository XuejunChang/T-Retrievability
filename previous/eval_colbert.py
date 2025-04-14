import sys
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
    retriever = pytcolbert.end_to_end() % retrieve_num

    csv = f'{nfs_dir}/{ranker}/df_{ranker}_{inx * 30}_{qstart//1000}k_{qend//1000}k_test5.csv'
    if os.path.exists(csv):
        print(f'{ranker} csv file {csv} already exists')
        df = pd.read_csv(csv, index_col=0).reset_index()
        return df
    else:
        print(f'start tramsforming {ranker} with threshold {threshold}')
        if qend == -1:
            qend = topics.shape[0]
        df = retriever.transform(topics[qstart:qend])
        print(f'df of {ranker} columns {df.columns.tolist()}')
        cols = ['qid', 'docid', 'docno', 'score', 'rank']
        print(f'to opt in columns {cols}')
        df = df[cols]
        df.to_csv(csv, index=False)
        print(f'saved {ranker} with shape {df.shape} into {csv}')

    return df

    # print('start calc {ranker} metrics')
    # mean, std_dev, gini_value = calc_stats(f'{ranker}', df, threshold, topics)
    # nDCG, RR = calc_irmetrics(df)
    # print(f'{ranker} mean: {mean}, std_dev: {std_dev}, gini_value:{gini_value}')
    # print(f'{ranker} nDCG: {nDCG}, RR: {RR}')
    #
    # result_df = pd.DataFrame([[mean, std_dev, gini_value, nDCG, RR]],columns=['mean', 'std_dev', 'gini_value', 'nDCG', 'RR'])
    # csv = f'{nfs_dir}/{ranker}/result_{ranker}_{inx*30}.csv'
    # result_df.to_csv(csv,index=False)
    # print(f'save to {csv}')

    # cache_bm25_dir = f"{nfs_dir}/bm25/bm25_cache_{str(threshold)}"
    # cached_bm25 = utils.get_cached_bm25(None, cache_bm25_dir)
    # pipeline2 = cached_bm25  % retrieve_num >> pt.text.get_text(dataset, 'text') >> pytcolbert.text_scorer()
    #
    # print('start tramsforming bm25 >> {ranker}')
    # df = pipeline2.transform(topics)
    # print('start calc bm25 >> {ranker} metrics')
    # mean, std_dev, gini_value = utils.calc_stats(f'bm25_{ranker}', df, threshold, topics)
    # nDCG, RR = utils.calc_irmetrics(df)
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

# qstart,qend = int(sys.argv[1]), int(sys.argv[2])
qstart,qend = 0,5
for i, threshold in enumerate(percent):
    if i == 1:
        evaluate_experiment(i, threshold, modelname, qstart, qend)

