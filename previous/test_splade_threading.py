import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

from Lz4PickleCache import *
import pandas as pd
import pyt_splade
splade = pyt_splade.Splade()


ranker = "splade"
retrieve_num = 100
dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()


threshold = 0.01

# Apply threshold to Splade query embeddings
def threshold_query(qdf):
    qdf["query_emb"] = qdf["query_emb"].apply(lambda x: {k: v for k, v in x.items() if v > threshold})
    return qdf

# Load the index
index_file = "/nfs/resources/cxj/retrievability-bias/splade/t5-base-msmarco-epoch-5-nostemmer-nostopwords-index--0.7423284530639649"

retriever = splade.query_encoder() >> pt.apply(threshold_query) >> pt.terrier.Retriever(index_file, wmodel='Tf', verbose=True)
results = retriever.transform(pd.DataFrame([{"query": "neural retrieval models"}]))
print(results)


