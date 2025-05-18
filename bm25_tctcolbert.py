import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, sys, pandas as pd
import fair_utils
import config
import tctcolbert
import bm25

def retrieve(modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_csv = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.csv'
    if not os.path.exists(result_csv):
        bm25_csv = bm25.retrieve(None, "bm25", dataset_name, topics_name, topics, retrieve_num, data_dir)
        bm25_df = pd.read_csv(bm25_csv,index_col=0).reset_index()
        bm25_df[['qid', 'docno']] = bm25_df[['qid', 'docno']].astype(str)

        dataset = pt.get_dataset(f'irds:{dataset_name}')
        pipeline = pt.text.get_text(dataset, 'text', verbose=True) >> tctcolbert.model
        print(f'tramsforming into {result_csv}')
        df = pipeline.transform(bm25_df)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid', 'docid', 'docno', 'score', 'rank']]

        print(f'saved into {result_csv}')
        df.to_csv(result_csv, index=False)
        print(f'done')

    return result_csv


# order of args: [version] retrieve
version = sys.argv[1]
data_dir = f'{config.data_dir}/{version}'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

run = sys.argv[2:]
modelname = "bm25_tctcolbert"

if __name__ == '__main__':
    if 'retrieve' in run:
        result_csv = retrieve(modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, data_dir)

        run_name=f'{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}'
        trec_res_path = fair_utils.save_trec_res(result_csv,run_name, data_dir)
        fair_utils.save_retrieved_docs_measures(result_csv, trec_res_path, data_dir)