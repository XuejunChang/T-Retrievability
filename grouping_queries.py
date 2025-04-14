import warnings

import faiss
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.started():
    pt.init()

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()

model = SentenceTransformer("msmarco-distilbert-base-v4")
query_embeddings = model.encode(topics["query"].tolist(), convert_to_numpy=True)
d = query_embeddings.shape[1]

num_clusters = [500, 1000, 2000, 5000, 10000]
# num_queries = [200,100,50,20,10]

for nc in num_clusters:
    print(f'clustering {nc}') # number of clusters(groups)
    # nq = total_topics/nc # the number of queries within each cluster(group)
    kmeans = faiss.Kmeans(d, nc, niter=100, verbose=True)
    kmeans.train(query_embeddings)

    # Assign each query to a cluster
    _, cluster_ids = kmeans.index.search(query_embeddings, 1)
    if 'cluster' in topics.columns:
        topics.drop(columns=['cluster'], inplace=True)

    topics["cluster"] = cluster_ids.flatten()

    # Retrieve queries from per cluster --- deprecated
    # sampled_queries = topics.groupby("cluster").progress_apply(lambda x: x.sample(n=min(nq, len(x)), random_state=42))
    
    csv = f'/nfs/primary/retrievability-bias/results/new_clustered/clustered_dev_queries_by_{nc}.csv'
    print(f'saving into {csv}')
    topics.to_csv(csv, index=False)
    print('done')
