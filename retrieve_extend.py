import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, sys
import config
import convert
import pyterrier_dr
from pyterrier_t5 import MonoT5ReRanker

def bm25_retrieve(index_path, modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    retriever = pt.terrier.Retriever(index_path, wmodel='BM25', verbose=True) % retrieve_num
    print(f'retrieving for {result_res_file}')
    df = retriever.transform(topics)
    print(f'df columns {df.columns.tolist()}')
    df = df[['qid','docid','docno','rank','score','query']]

    # result_res_file = f'{os.path.splitext(result_res_file)[0]}.csv'
    # df.to_csv(result_res_file, index=False)

    result_res_file = f'{os.path.splitext(result_res_file)[0]}.parquet'
    print(f'saving {result_res_file}')
    df.to_parquet(result_res_file)
    result_res_file = f'{os.path.splitext(result_res_file)[0]}.res'
    run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
    convert.convert_docdf2_trec(df, result_res_file, run_name)

    return result_res_file

def tctcolbert_retrieve(index_path, modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        index = pyterrier_dr.FlexIndex(index_path)
        model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco', batch_size=16, verbose=True)
        retriever = model >> index.np_retriever() % retrieve_num
        print(f'tramsforming into {result_res_file}')
        df = retriever.transform(topics)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid', 'docid', 'docno','rank', 'score']]
        df.to_csv(f'{os.path.splitext(result_res_file)[0]}.csv', index=False)

        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        convert.save_trec_res(df, result_res_file, run_name)

    return result_res_file

def bm25_tctcolbert_retrieve(modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        bm25_res = bm25_retrieve(None, "bm25", dataset_name, topics_name, topics, retrieve_num, data_dir)
        bm25_df = convert.convert_res2df(bm25_res)
        model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco', batch_size=16, verbose=True)
        pipeline = pt.text.get_text(config.dataset, 'text', verbose=True) >> model

        print(f'tramsforming into {result_res_file}')
        df = pipeline.transform(bm25_df)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid', 'docid', 'docno','rank', 'score']]
        df.to_csv(f'{os.path.splitext(result_res_file)[0]}.csv', index=False)

        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        convert.save_trec_res(df, result_res_file, run_name)

    return result_res_file

def bm25_monot5_retrieve(modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        bm25_res = bm25_retrieve(None, "bm25", dataset_name, topics_name, topics, retrieve_num, data_dir)
        bm25_df = convert.convert_res2df(bm25_res)
        monot5 = pt.text.get_text(config.dataset, 'text', verbose=True) >> MonoT5ReRanker(batch_size=16)

        print(f'tramsforming into {result_res_file}')
        df = monot5.transform(bm25_df)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid', 'docid', 'docno','rank', 'score']]
        df.to_csv(f'{os.path.splitext(result_res_file)[0]}.csv', index=False)

        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        convert.save_trec_res(df, result_res_file, run_name)

    return result_res_file

modelname = sys.argv[1]

if __name__ == '__main__':
    if modelname == 'bm25':
        index_path = f"{config.data_dir}/{modelname}-{config.dataset_name}-index"
        result_res_file = bm25_retrieve(index_path, modelname, config.dataset_name, config.train_topics_name, config.train_topics, config.retrieve_num, config.data_dir)

    if modelname == 'tctcolbert':
        index_path = f"{config.data_dir}/{modelname}-{config.dataset_name}-index.flex"
        result_res_file = tctcolbert_retrieve(index_path, modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)
    
    if modelname == 'bm25_tctcolbert':
        result_res_file = bm25_tctcolbert_retrieve(modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)

    if modelname == 'bm25_monot5':
        result_res_file = bm25_monot5_retrieve(modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)

