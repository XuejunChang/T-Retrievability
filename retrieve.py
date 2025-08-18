import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, sys, pandas as pd
import fair_utils
import config
import pyt_splade
import pyterrier_dr
from pyterrier_t5 import MonoT5ReRanker

def bm25_retrieve(index_path, modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        retriever = pt.terrier.Retriever(index_path, wmodel='BM25', verbose=True) % retrieve_num
        print(f'tramsforming into dataframe')
        df = retriever.transform(topics)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid','docid','docno','score','rank','query']]
        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        fair_utils.save_trec_res(df, result_res_file, run_name)

    return result_res_file

def splade_retrieve(index_path, modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        model = pyt_splade.Splade()
        retriever = model.query_encoder() >> pt.terrier.Retriever(index_path, wmodel='Tf', verbose=True) % retrieve_num
        print(f'tramsforming into dataframe')
        df = retriever.transform(topics)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid','docid','docno','score','rank']]

        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        fair_utils.save_trec_res(df, result_res_file, run_name)

    return result_res_file

def tctcolbert_retrieve(index_path, modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        index = pyterrier_dr.FlexIndex(index_path)
        model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco', batch_size=16, verbose=True)
        retriever = model >> index.np_retriever() % retrieve_num
        print(f'tramsforming into dataframe')
        df = retriever.transform(topics)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid','docid','docno','score','rank']]

        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        fair_utils.save_trec_res(df, result_res_file, run_name)

    return result_res_file

def bm25_tctcolbert_retrieve(modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        bm25_csv = bm25_retrieve(None, "bm25", dataset_name, topics_name, topics, retrieve_num, data_dir)
        bm25_df = pd.read_csv(bm25_csv,index_col=0).reset_index()
        bm25_df[['qid', 'docno']] = bm25_df[['qid', 'docno']].astype(str)

        dataset = pt.get_dataset(f'irds:{dataset_name}')
        model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco', batch_size=16, verbose=True)
        pipeline = pt.text.get_text(dataset, 'text', verbose=True) >> model
        print(f'tramsforming into dataframe')
        df = pipeline.transform(bm25_df)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid', 'docid', 'docno', 'score', 'rank']]

        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        fair_utils.save_trec_res(df, result_res_file, run_name)

    return result_res_file

def bm25_monot5_retrieve(modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        bm25_csv = bm25_retrieve(None, "bm25", dataset_name, topics_name, topics, retrieve_num, data_dir)
        bm25_df = pd.read_csv(bm25_csv,index_col=0).reset_index()
        bm25_df[['qid', 'docno']] = bm25_df[['qid', 'docno']].astype(str)
        dataset = pt.get_dataset(f'irds:{dataset_name}')
        monot5 = pt.text.get_text(dataset, 'text', verbose=True) >> MonoT5ReRanker(batch_size=16)

        print(f'tramsforming into dataframe')
        df = monot5.transform(bm25_df)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid', 'docid', 'docno', 'score', 'rank']]

        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        fair_utils.save_trec_res(df, result_res_file, run_name)

    return result_res_file

modelname = sys.argv[1]
if __name__ == '__main__':
    if modelname == 'bm25':
        index_path = f"{config.data_dir}/{modelname}-{config.dataset_name}-nostemmer-nostopwords-index"
        result_res_file = bm25_retrieve(index_path, modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)

    if modelname == 'splade':
        index_path = f"{config.data_dir}/{modelname}-{config.dataset_name}-nostemmer-nostopwords-index"
        result_res_file = splade_retrieve(index_path, modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)

    if modelname == 'tctcolbert':
        index_path = f"{config.data_dir}/{modelname}-{config.dataset_name}-index.flex"
        result_res_file = tctcolbert_retrieve(index_path, modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)

    if modelname == 'bm25_tctcolbert':
        result_res_file = bm25_tctcolbert_retrieve(modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)

    if modelname == 'bm25_monot5':
        result_res_file = bm25_monot5_retrieve(modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)

