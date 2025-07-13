import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, sys, pandas as pd
import fair_utils
import config
import pyt_splade

def splade_retrieve(index_path, modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_res_file = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.res'
    if os.path.exists(result_res_file):
        print(f'found {result_res_file}')
    else:
        model = pyt_splade.Splade()
        retriever = model.query_encoder() >> pt.terrier.Retriever(index_path, wmodel='Tf', verbose=True) % retrieve_num
        print(f'tramsforming into {result_res_file}')
        df = retriever.transform(topics)
        print(f'df columns {df.columns.tolist()}')

        df = df[['qid','docid','docno','rank','score']]
        df.to_csv(f'{os.path.splitext(result_res_file)[0]}.csv', index=False)

        run_name = f'{modelname}_{dataset_name}_{topics_name}_{retrieve_num}'
        fair_utils.save_trec_res(df, result_res_file, run_name)

    return result_res_file


modelname = sys.argv[1]

# nohup python -u retrieve.py v1 bm25 > ./logbm25rtr200 2>&1 &
if __name__ == '__main__':

    if modelname == 'splade':
        index_path = f"{config.index_dir}/{modelname}-{config.dataset_name}-nostemmer-nostopwords-index"
        result_res_file = splade_retrieve(index_path, modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, config.data_dir)

