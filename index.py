import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import sys
import config
import pyt_splade
import pyterrier_dr

def create_bm25_index(index_path, dataset):
    print(f"indexing into {index_path}")
    indexer = pt.IterDictIndexer(index_path,stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, verbose=True)
    indexref = indexer.index(dataset.get_corpus_iter(verbose=True))

    print('indexing done')
    return indexref

def create_splade_index(index_path, dataset):
    print(f"indexing into {index_path}")
    pipe = model.doc_encoder() >> pt.IterDictIndexer(index_path, stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, pretokenised=True, verbose=True)
    pipe.index(dataset.get_corpus_iter(verbose=True))

    return index_path

def create_tctcolbert_index(index_path, dataset):
    print(f"indexing into {index_path}")
    index = pyterrier_dr.FlexIndex(f'{index_path}')
    pipeline =  model >> index
    pipeline.index(dataset.get_corpus_iter(verbose=True))
    print(f'indexing done')

    return index_path

modelname = sys.argv[1]
if __name__ == '__main__':
    if modelname == 'bm25':
        index_path = f"{config.data_dir}/{modelname}-{config.dataset_name}-nostemmer-nostopwords-index"
        create_bm25_index(index_path, config.dataset)

    if modelname == 'splade':
        model = pyt_splade.Splade()
        index_path = f"{config.data_dir}/{modelname}-{config.dataset_name}-nostemmer-nostopwords-index"
        create_splade_index(index_path, config.dataset)

    if modelname == 'tctcolbert':
        model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco', batch_size=16, verbose=True)
        index_path = f"{config.data_dir}/{modelname}-{config.dataset_name}-index.flex"
        create_tctcolbert_index(index_path, config.dataset)


