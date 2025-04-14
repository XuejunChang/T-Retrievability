import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os

import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()

import datetime
import os
import sys
import numpy as np
import torch
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

# model_name = "facebook/contriever"  # Contriever model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "facebook/contriever-msmarco"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)  # Move model to GPU

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os

# Parameters
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "facebook/contriever-msmarco"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)  # Move model to GPU


shard_size = 10  # Number of documents per shard
embedding_dim = 768  # Embedding dimension (Contriever-specific)
embedding_dir = "sharded_embeddings"
os.makedirs(embedding_dir, exist_ok=True)

# Load the dataframe with 8.8 million documents
df = pd.DataFrame(dataset.get_corpus_iter(verbose=True))[:100]

# Function to generate embeddings for a batch of texts
def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Example: mean pooling


# Process the DataFrame in chunks (shards)
for i in range(0, len(df), shard_size):
    shard_texts = df['text'].iloc[i:i + shard_size].tolist()  # Get text from shard
    shard_embeddings = generate_embeddings(shard_texts)

    # Save shard embeddings to disk
    shard_file = os.path.join(embedding_dir, f"shard_{i}.npy")
    np.save(shard_file, shard_embeddings)
    print(f"Saved embeddings for shard {i} to {shard_file}")

import faiss

# Parameters
nlist = 10  # Number of clusters for IVF index
faiss_index_dir = "faiss_indexes"
os.makedirs(faiss_index_dir, exist_ok=True)

# Create FAISS index for each shard
for shard_file in sorted(os.listdir(embedding_dir)):
    if shard_file.endswith(".npy"):
        # Load shard embeddings
        shard_embeddings = np.load(os.path.join(embedding_dir, shard_file))

        # Initialize FAISS index
        quantizer = faiss.IndexFlatL2(embedding_dim)  # Exact search quantizer
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)

        # Train the index on the shard embeddings
        if not index.is_trained:
            index.train(shard_embeddings)

        # Add shard embeddings to the index
        index.add(shard_embeddings)

        # Save the index to disk
        shard_index_file = os.path.join(faiss_index_dir, shard_file.replace(".npy", ".index"))
        faiss.write_index(index, shard_index_file)
        print(f"Saved FAISS index for shard {shard_file} to {shard_index_file}")

# Generate query embedding
query_text = topics.loc[0,'query']
query_inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    query_embedding = model(**query_inputs).last_hidden_state.mean(dim=1).cpu().numpy()

# Search across all shards
k = 2  # Top 100 results
all_distances, all_indices = [], []

for index_file in sorted(os.listdir(faiss_index_dir)):
    if index_file.endswith(".index"):
        # Load the FAISS index
        index = faiss.read_index(os.path.join(faiss_index_dir, index_file))

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

print("Top 100 document indices:", final_indices)
print("Top 100 distances:", final_distances)
