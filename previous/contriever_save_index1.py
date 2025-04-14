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
df = pd.DataFrame(dataset.get_corpus_iter(verbose=True))

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

            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

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

# save_embeddings()
# create_index()

# Search across all shards
k = 100  # Top 100 results
batch_size = 1000
res = []
for j in range(0, topics.shape[0], batch_size):
    print(f'{datetime.datetime.now()} start ----q:{j}')
    queries = topics[j:j + batch_size]['query'].to_list()

    print(f'{datetime.datetime.now()} start calc queries embeddings')
    query_embedding = normalize(generate_embeddings(queries, batch_size=batch_size))

    # retrieve top 100 docs for this query.
    all_distances, all_indices = [], []
    for index_file in sorted(os.listdir(faiss_index_dir)):
        if index_file.endswith(".index"):
            # Load the FAISS index
            index = faiss.read_index(os.path.join(faiss_index_dir, index_file))

            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

            # Search the index
            distances, indices = index.search(query_embedding, k)
            all_distances.append(distances)
            all_indices.append(indices)

    # Combine results from all shards
    all_distances = np.hstack(all_distances)  # Combine distances
    all_indices = np.hstack(all_indices)  # Combine indices

    # Sort to get the top 100 results globally
    sorted_indices = np.argsort(all_distances[0])[:k]
    final_indices = all_indices[0][sorted_indices]
    final_distances = all_distances[0][sorted_indices]

    # print("Top 100 document indices:", final_indices)
    # print("Top 100 distances:", final_distances)

    qid = topics.loc[j, 'qid']
    for rank, (distance, idx) in enumerate(zip(final_distances, final_indices)):
        docno = df.loc[idx, 'docno']
        # the doc is retrieved 1 time
        res.append([qid, docno, rank, distance])

# res_df = pd.DataFrame(res, columns=['qid', 'docno', 'rank', 'distance'])
# print('start saving to csv file')
# csvfile = f'/nfs/datasets/cxj/retrievability-bias/results_contriever_100_IndexFlatL2.csv'
# if os.path.exists(csvfile):
#     os.remove(csvfile)
#     print(f'{csvfile} deleted')
#
# res_df.to_csv(csvfile, index=False)
# print(f'{csvfile} saved')
