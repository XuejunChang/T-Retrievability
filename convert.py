import numpy as np, pandas as pd
import os,sys, glob
import config

def convert_docdf2_trec(df, res_file_path, run_name):
    if os.path.exists(res_file_path):
        print(f'Exists {res_file_path}')
        os.remove(res_file_path)
        print(f'removed {res_file_path}')

    result = pd.DataFrame()
    result['query_id'] = df['qid']
    result['Q0'] = 'Q0'
    result['doc_id'] = df['docid']
    result['rank'] = df['rank']
    result['score'] = df['score']
    result['run_name'] = run_name

    print(f'saving into {res_file_path}')
    result.to_csv(res_file_path, sep=' ', index=False, header=False)
    print(f'saved')

def convert_dfall2trec(df, file_path):
    if os.path.exists(file_path):
        print(f'Exists {file_path}')
        os.remove(file_path)
        print(f'removed {file_path}')

    print(f'saving into {file_path}')
    df.to_csv(file_path, sep=' ', index=False, header=False) # qid query
    print(f'saved')

    return file_path

def batch_convert_df2trec(query_dir):
    files = glob.glob(f'{query_dir}/*.csv')
    for query_csv_path in files:
        print(f'processing {query_csv_path}')
        df = pd.read_csv(query_csv_path, index_col=0).reset_index()
        query_res_path = os.path.splitext(query_csv_path)[0] + '.res'
        convert_dfall2trec(df, query_res_path)

def convert_res2docdf(res_file_path, columns=None):
    print(f'converting {res_file_path} into a dataframe')
    df = pd.read_csv(res_file_path, sep=r"\s+", names=columns)
    print('done')
    return df

def convert_qrels2_trec(df, qrels_res_path):
    if os.path.exists(qrels_res_path):
        print(f'Exists {qrels_res_path}')
        os.remove(qrels_res_path)
        print(f'removed {qrels_res_path}')

    result = pd.DataFrame()
    result['query_id'] = df['qid']
    result['Q0'] = 0
    result['doc_id'] = df['docno']
    result['relevance'] = df['label']

    print(f'saving into {qrels_res_path}')
    result.to_csv(qrels_res_path, sep=' ', index=False, header=False)
    print(f'saved')

if __name__ == '__main__':
    for modelname in config.models:
        csv = f'{config.data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}.csv'
        df = pd.read_csv(csv, index_col=0).reset_index()
        run_name = os.path.splitext(csv)[0].split('/')[-1]
        res = os.path.splitext(csv)[0] + '.res'
        convert_docdf2_trec(df, res, run_name)

    # qrels_res_path = f'{config.data_dir}/qrels_dev.res'
    # qrels_path = convert_qrels2_trec(config.qrels, qrels_res_path)




