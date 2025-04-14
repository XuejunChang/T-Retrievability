import sys
import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()
import faiss
import datetime
import os
import numpy as np
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

from transformers import AutoTokenizer, AutoModel
import torch


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "facebook/contriever-msmarco"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)  # Move model to GPU


shard_size = 1000  # Number of documents per shard
embedding_dim = 768  # Embedding dimension (Contriever-specific)
embedding_dir = f"{nfs_save}/contriever/sharded_embeddings"
os.makedirs(embedding_dir, exist_ok=True)


# Function to generate embeddings for a batch of texts
# def generate_embeddings(texts):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Example: mean pooling

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

# Process the DataFrame in chunks (shards)
# df = pd.DataFrame(dataset.get_corpus_iter(verbose=True))
def save_embeddings():
    for i in range(0, len(df), shard_size):
        print(f"start embeddings for shard {i}")
        shard_texts = df['text'].iloc[i:i + shard_size].tolist()  # Get text from shard
        shard_embeddings = normalize(generate_embeddings(shard_texts))

        # Save shard embeddings to disk
        shard_file = os.path.join(embedding_dir, f"shard_{i}.npy")
        np.save(shard_file, shard_embeddings)
        print(f"Saved embeddings for shard {i} to {shard_file}")



# nlist = 20  # Number of clusters for IVF index
faiss_index_dir = f"{nfs_save}/contriever/faiss_IndexFlatL2"
os.makedirs(faiss_index_dir, exist_ok=True)

# Create FAISS index for each shard

def create_index():
    for shard_file in sorted(os.listdir(embedding_dir)):
        if shard_file.endswith(".npy"):
            # Load shard embeddings
            shard_embeddings = np.load(os.path.join(embedding_dir, shard_file))

            index = faiss.IndexFlatL2(embedding_dim)

            # quantizer = faiss.IndexFlatL2(embedding_dim)  # Exact search quantizer
            # index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)

            gpu = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu, 0, index)

            # Train the index on the shard embeddings
            if not index.is_trained:
                index.train(shard_embeddings)

            # Add shard embeddings to the index
            index.add(shard_embeddings)
            cpu_index = faiss.index_gpu_to_cpu(index)
            # Save the index to disk
            shard_index_file = os.path.join(faiss_index_dir, shard_file.replace(".npy", ".index"))

            faiss.write_index(cpu_index, shard_index_file)
            print(f"Saved FAISS index for shard {shard_file} to {shard_index_file}")

import heapq
def merge_shard_results(all_results, top_k):
    num_queries = all_results[0][0].shape[0]
    merged_distances = []
    merged_indices = []

    for q in range(num_queries):
        heap = []
        for shard_id, (distances, indices) in enumerate(all_results):
            for i in range(top_k):
                heapq.heappush(heap, (distances[q, i], (shard_id, indices[q, i])))

        # Extract top_k results
        top_results = heapq.nsmallest(top_k, heap)
        merged_distances.append([x[0] for x in top_results])
        merged_indices.append([x[1] for x in top_results])

    return np.array(merged_distances), np.array(merged_indices)

# save_embeddings()
# create_index()

# Search across all shards
k = 100  # Top 100 results
batch_size = int(sys.argv[1])
print(f'batch size=: {batch_size}`')

for j in range(0, topics.shape[0], batch_size):
    print(f'{datetime.datetime.now()} start ----q:{j}')
    queries = topics[j:j + batch_size]['query'].to_list()

    print(f'{datetime.datetime.now()} start calc queries embeddings')
    query_embedding = normalize(generate_embeddings(queries, batch_size=batch_size))

    shard_paths = [f'{faiss_index_dir}/{index_file}' for index_file in os.listdir(faiss_index_dir) if index_file.endswith(".index")]
    # Load shards
    shards = [faiss.read_index(path) for path in shard_paths]

    all_results = []
    for i, shard in enumerate(shards):
        gpu = faiss.StandardGpuResources()
        shard = faiss.index_cpu_to_gpu(gpu, 0, shard)
        distances, indices = shard.search(query_embedding, k)
        all_results.append((distances, indices))

    merged_distances, merged_indices = merge_shard_results(all_results, k)

    offsets = [0] + np.cumsum([shard.ntotal for shard in shards]).tolist()
    # Convert shard-specific IDs to global document IDs
    global_indices = []
    for q in range(len(queries)):
        global_indices.append([
            offsets[shard_id] + local_id
            for shard_id, local_id in merged_indices[q]
        ])
    res = []
    for qid, (distances, doc_ids) in enumerate(zip(merged_distances, global_indices)):
        for rank, (distance, did) in enumerate(zip(distances, doc_ids)):
            res.append({
                "qid": qid,
                "did": did,
                "rank": rank,
                "distance": distance
            })
    # print(res)

    res_df = pd.DataFrame(res, columns=["qid", "did", "rank", "distance"])
    print('start saving to csv file')
    csvfile = f'/nfs/datasets/cxj/retrievability-bias/contriever/batches/results_contriever_100_qbatch_{j}.csv'
    if os.path.exists(csvfile):
        os.remove(csvfile)
        print(f'{csvfile} deleted')

    res_df.to_csv(csvfile, index=False)
    print(f'{csvfile} saved')
    del res_df
    del res


