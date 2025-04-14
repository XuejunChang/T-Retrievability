import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from Lz4PickleCache import *
import pandas as pd
import os
import numpy as np
import pyt_splade
splade = pyt_splade.Splade()

root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'

ranker = "splade"
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

# def threshold_docs(docs_df, threshold=0.01):
#     docs_df["doc_rep"] = docs_df["doc_rep"].apply(lambda x: {k: v for k, v in x.items() if v > threshold})
#     return docs_df
#

# documents = pd.DataFrame([...])
# documents = threshold_docs(documents, threshold=0.01)
#
# indexer = pt.IterDictIndexer("./splade_index")
# indexref = indexer.index(documents.to_dict(orient="records"))
pt.tqdm.pandas()
def threshold_query(q_df):
    q_df["query_rep"] = q_df["query_rep"].progress_apply(lambda x: {k: v for k, v in x.items() if v > threshold})
    return q_df

def evaluate_experiment(inx, threshold, modelname):
    index_file = f"{nfs_dir}/{ranker}/{modelname}-nostemmer-nostopwords-index-{str(threshold)}"
    if not os.path.exists(index_file):
        print(f"Index file {index_file} does not exist")
        return
    # cache_dir = f"{nfs_dir}/{ranker}/{ranker}_cache_{str(threshold)}"
    # cached_splade = utils.get_cached_splade(index_file, cache_dir,splade)

    retriever = splade.query_encoder() >> pt.apply(threshold_query) >> pt.terrier.Retriever(index_file, wmodel='Tf',
                                                                                            verbose=True)

    retriever = retriever % retrieve_num

    print(f'start tramsforming {ranker}')
    csv = f'{nfs_dir}/{ranker}/thresholds/df_{ranker}_{inx * 30}.csv'
    if os.path.exists(csv):
        df = pd.read_csv(csv,index_col=1).reset_index()
    else:
        df = retriever.transform(topics)
        print(f'df of {ranker} columns {df.columns.tolist()}')
        df = df[['qid','docid','docno','score','rank']]
        print(f'to to opt in in df of {ranker} shape {df.shape}')
        df.to_csv(csv,index=False)
        print(f'save {ranker} with shape {df.shape} into {csv}')

    # print('start calc {ranker} metrics')
    # mean, std_dev, gini_value = utils.calc_stats(f'{ranker}', df, threshold, topics)
    # nDCG, RR = utils.calc_irmetrics(df)
    # print(f'{ranker} mean: {mean}, std_dev: {std_dev}, gini_value:{gini_value}')
    # print(f'{ranker} nDCG: {nDCG}, RR: {RR}')
    #
    # result_df = pd.DataFrame([[mean, std_dev, gini_value, nDCG, RR]],columns=['mean', 'std_dev', 'gini_value', 'nDCG', 'RR'])
    # csv = f'{nfs_dir}/{ranker}/result_{ranker}_{inx*30}.csv'
    # result_df.to_csv(csv,index=False)
    # print(f'save to {csv}')

def get_percentage(cache_file, qual_signal=None):
    print('calculating percentage...')
    signal = np.array([p[qual_signal] for p in pt.tqdm(cache_file.get_corpus_iter())])
    percent = np.nanpercentile(signal, [x for x in range(0, 101, 5)])
    return percent

modelname = 't5-base-msmarco-epoch-5'
cache_file = Lz4PickleCache(f'/nfs/resources/cxj/retrievability-bias/supervisedT5/npy121pruned/{modelname}.lz4')

pert_file = f'{nfs_dir}/bm25/percentage.npy'
if os.path.exists(pert_file):
    percent = np.load(pert_file)
else:
    percent = get_percentage(cache_file, qual_signal="prob")
    percent = np.array(percent[:-1]).take([0, 30/5, 60/5,90/5])  # take 30%, 60%, 90% pruned
    np.save(pert_file, percent)

for i, threshold in enumerate(percent):
    if i == 0:
        continue
    evaluate_experiment(i, threshold, modelname)
