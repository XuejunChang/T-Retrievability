import warnings

warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()

import os
import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', False)
import pyt_splade
splade = pyt_splade.Splade()

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

work_name = "retrievability-bias"
root_dir = f'/root/{work_name}'
nfs_save = f'/nfs/datasets/cxj/{work_name}'
if not os.path.exists(nfs_save):
    os.makedirs(nfs_save)

index_name = "msmarco-passage-splade-nostemmer-nostopwords-index"

def create_indexes():
    index_file = f"{root_dir}/{index_name}"
    nfs_index_file = f"{nfs_save}/{index_name}"

    if not os.path.exists(nfs_index_file):
        print(f"indexing into {index_file}")
        # pipe = splade.doc_encoder() >> pt.IterDictIndexer(index_file, stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, blocks=True, pretokenised=True, verbose=True)
        pipe = splade.doc_encoder() >> pt.IterDictIndexer(index_file, stemmer=pt.TerrierStemmer.none,
                                                          stopwords=pt.TerrierStemmer.none, pretokenised=True,
                                                          verbose=True)
        pipe.index(dataset.get_corpus_iter(verbose=True), batch_size=16)
        os.system(f'cp -r {index_file} {nfs_save}/')
        print(f'copied index into {nfs_save}')

    return nfs_index_file




if __name__ == '__main__':
    index = create_indexes()
    br = splade.query_encoder() >> pt.terrier.Retriever(index, wmodel='Tf', verbose=True) % 100
    pipe = br >> pt.text.get_text(dataset, 'text')
    print('start splade retrieval')
    result = pipe.transform(topics)

    csv = f'results_splade_100.csv'
    result.to_csv(f'{root_dir}/{csv}', index=False)
    os.system(f'cp -r {root_dir}/{csv} {nfs_save}/')
    print(f'copied {root_dir}/{csv} into {nfs_save}')