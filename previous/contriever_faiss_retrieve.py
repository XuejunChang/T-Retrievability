import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()

import os
import sys
import numpy as np
import torch
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', False)
import datetime

dataset = pt.get_dataset(f'irds:msmarco-passage')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/dev')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

work_name = "retrievability-bias"
root_dir = f'/root/{work_name}'
nfs_save = f'/nfs/datasets/cxj/{work_name}'
if not os.path.exists(nfs_save):
    os.makedirs(nfs_save)

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import faiss
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and tokenizer
model_name = "facebook/contriever-msmarco"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)  # Move model to GPU

# def calc_embeddings(inputs):
#     # Tokenize the inputs
#     tokenized_inputs = tokenizer(inputs,padding=True,truncation=True,return_tensors="pt").to(device)  # Move tokenized inputs to GPU
#
#     # Compute embeddings
#     with torch.no_grad():
#         token_embeddings = model(**tokenized_inputs).last_hidden_state
#
#     # Aggregate embeddings (e.g., mean pooling)
#     attention_mask = tokenized_inputs["attention_mask"]  # To ignore padding tokens in the aggregation
#     masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # Apply attention mask
#     sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over the sequence length
#     sum_mask = attention_mask.sum(dim=1).unsqueeze(-1)  # Count non-padding tokens per sequence
#     inputs_embeddings = sum_embeddings / sum_mask  # Mean pooling: divide by token counts
#
#     return inputs_embeddings

def generate_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            # Use pooler_output for embeddings (alternatively, sum CLS token or hidden states)
            batch_embeddings = model(**inputs).pooler_output
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

# Normalize embeddings
def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

documents = [
    "This is a sample document.",
    "Another example sentence.",
    "More data for testing purposes.",
]

queries = [
    "This is a sample document.",
    "Another example sentence.",
    "More data for testing purposes.",
]

# Faiss index
embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for Cosine Similarity

res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)

index.add(doc_embeddings)


doc_ids = ["doc1", "doc2", "doc3","doc1", "doc2", "doc3","doc1", "doc2", "doc3"]
for query_idx, (distances, indices) in enumerate(zip(D, I)):
    print(f"Query Document: {doc_ids[query_idx]}")
    for rank, (distance, idx) in enumerate(zip(distances, indices)):
        print(f"  Rank {rank + 1}: DocID = {doc_ids[idx]}, Similarity = {distance:.4f}")


df = pd.DataFrame(dataset.get_corpus_iter(verbose=True))
d1 = int(sys.argv[1])
d2 = int(sys.argv[2])
if d2 == -1:
    d2 = df.shape[0]

# q1 = int(sys.argv[3])
# q2 = int(sys.argv[4])
# if q2 == -1:
#     q2 = topics.shape[0]

print(f'd1={d1},d2={d2}')
batch_size = 1000
k = 100
q_emb_dict = {}
for i in range(d1,d2,batch_size):
    documents = df[i:i+batch_size]['text'].to_list()
    print(f'{datetime.datetime.now()} start calc documents embeddings')
    doc_embeddings = normalize(generate_embeddings(documents))
    print(f'{datetime.datetime.now()} done')

    res = []
    for j in range(0,topics.shape[0], batch_size):
        print(f'{datetime.datetime.now()} start ----d:{i} -----q:{j}')
        queries = topics[j:j+batch_size]['query'].to_list()

        if str(j) in q_emb_dict:
            queries_cpu = q_emb_dict[str(j)]
            print(f'{datetime.datetime.now()} loaded queries embeddings from memory')
        else:
            print(f'{datetime.datetime.now()} start calc queries embeddings')
            query_embeddings = normalize(generate_embeddings(queries))
            print(f'{datetime.datetime.now()} done')


        print(f'start computing cosine similarity')
        D, I = index.search(query_embeddings, k)
        print('start appending results into a res list')
        for query_idx, (distances, indices) in enumerate(zip(D, I)):
            qid = topics.loc[j + query_idx, 'qid']
            # print(f"query {qid}:")
            for rank, (distance, idx) in enumerate(zip(distances, indices)):
                docno = df.loc[i + idx, 'docno']
                # the doc is retrieved 1 time
                res.append([qid, docno, rank, distance])


    res_df = pd.DataFrame(res, columns=['qid','docno','rtr_cnt','sim_score'])
    print('start saving to csv file')
    csvfile = f'/nfs/llm/cxj_models/retr_bias/contriever_msmarco_{i}.csv'
    if os.path.exists(csvfile):
        os.remove(csvfile)
        print(f'{csvfile} deleted')

    res_df.to_csv(csvfile, index=False)
    print(f'{csvfile} saved')

    del document_embeddings
    res = []
    res_df = None
    torch.cuda.empty_cache()



