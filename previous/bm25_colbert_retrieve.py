import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from pyterrier_colbert.ranking import ColBERTFactory
import os
import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', False)

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

work_name = "retrievability-bias"
root_dir = f'/root/{work_name}'
nfs_save = f'/nfs/datasets/cxj/{work_name}'
if not os.path.exists(nfs_save):
    os.makedirs(nfs_save)

from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory

def create_lexical_indexes():
    index_file = f"{root_dir}/msmarco-passage-nostemmer-nostopwords-index"
    nfs_index_file = f"{nfs_save}/msmarco-passage-nostemmer-nostopwords-index"

    if not os.path.exists(nfs_index_file):
        print(f"indexing into {index_file}")
        indexer = pt.IterDictIndexer(index_file, stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, verbose=True)
        indexer.index(dataset.get_corpus_iter(verbose=True))
        os.system(f'cp -r {index_file} {nfs_save}/')
        print(f'copied index into {nfs_save}')

    return nfs_index_file

# CHECKPOINT = "/nfs/primary/data/llm/colbert_v2/colbert.dnn"
CHECKPOINT="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
def create_colbert_indexes():
    col_index = f"{root_dir}/msmarco-passage-colbert-index"
    nfs_index_file = f"{nfs_save}/msmarco-passage-colbert-index"

    if not os.path.exists(nfs_index_file):
        print(f"start indexing into {col_index}")
        indexer = ColBERTIndexer(CHECKPOINT, f"{root_dir}/", "msmarco-passage-colbert-index", chunksize=8, gpu=True)
        indexer.index(dataset.get_corpus_iter(verbose=True))
        os.system(f'cp -r {root_dir}/msmarco-passage-colbert-index {nfs_save}/')
        print(f'copied {root_dir}/msmarco-passage-colbert-index into {nfs_save}')

    return nfs_index_file


lexical_index = create_lexical_indexes()
bm25 = pt.BatchRetrieve(lexical_index, wmodel='BM25', verbose=True) % 100

# Load ColBERT model
colbert_index = create_colbert_indexes().split('/')[-1]
colbert = ColBERTFactory(CHECKPOINT, f"{root_dir}/", f"{colbert_index}")

# Create pipeline
pipeline = bm25 >> pt.text.get_text(dataset,'text') >> colbert.text_scorer()

results = pipeline(topics)
csv = f'results_bm25_colbert_100.csv'
results.to_csv(f'{root_dir}/{csv}')
os.system(f'cp -r {root_dir}/{csv} {nfs_save}/')
print(f'copied {root_dir}/{csv} into {nfs_save}')

