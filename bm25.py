import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, sys
import fair_utils
import config

def create_index(index_path, dataset):
    print(f"indexing into {index_path}")
    indexer = pt.IterDictIndexer(index_path,stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, verbose=True)
    indexref = indexer.index(dataset.get_corpus_iter(verbose=True))

    print('indexing done')
    return indexref

def retrieve(index_path, modelname, dataset_name, topics_name, topics, retrieve_num, data_dir):
    result_pkl = f'{data_dir}/{modelname}_{dataset_name}_{topics_name}_{retrieve_num}.pkl'
    if not os.path.exists(result_pkl):
        retriever = pt.terrier.Retriever(index_path, wmodel='BM25', verbose=True) % retrieve_num
        print(f'tramsforming into {result_pkl}')
        df = retriever.transform(topics)
        print(f'df columns {df.columns.tolist()}')
        df = df[['qid','docid','docno','score','rank','query']]

        print(f'saved into {result_pkl}')
        df.to_pickle(result_pkl)
        print(f'done')

    return result_pkl

# order of args: [version] retrieve
version = sys.argv[1]
data_dir = f'{config.data_dir}/{version}'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

run = sys.argv[2:]
modelname = "bm25"
index_path = f"{data_dir}/{modelname}-{config.dataset_name}-nostemmer-nostopwords-index"

if __name__ == '__main__':
    if 'index' in run:
        create_index(index_path, config.dataset)

    if 'retrieve' in run:
        result_pkl = retrieve(index_path, modelname, config.dataset_name, config.topics_name, config.topics, config.retrieve_num, data_dir)

        run_name=f'{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}'
        trec_res_path = fair_utils.save_trec_res(result_pkl,run_name, data_dir)
        fair_utils.save_retrieved_docs_measures(result_pkl, trec_res_path, data_dir)



