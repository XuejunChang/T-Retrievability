import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.java.started():
    pt.java.init()

import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', False)


dataset_name = 'msmarco-passage'
eval_ds_name = 'dev'
dataset = pt.get_dataset(f'irds:{dataset_name}')
eval_dataset = pt.get_dataset(f'irds:{dataset_name}/{eval_ds_name}')
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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and tokenizer
model_name = "facebook/contriever-msmarco"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)  # Move model to GPU

# Function to encode text into dense vectors
def encode_texts(texts):
    # Tokenize and move input tensors to GPU
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        # Generate embeddings and move them back to CPU for further processing
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
    return embeddings

# Define query and documents
query = "What is the capital of France?"
documents = [
    "Paris is the capital city of France.",
    "France is a country in Europe.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Paris is known for the Eiffel Tower.",
    "The Louvre is located in Paris, France.",
    "London is the capital of the United Kingdom.",
    "Rome is the capital of Italy.",
    "Paris is famous for its cuisine.",
    "The French language is spoken in Paris.",
]

# Encode query and documents
query_embedding = encode_texts([query])  # Query embedding
document_embeddings = encode_texts(documents)  # Document embeddings

# Compute cosine similarity
cos_sim = cosine_similarity(query_embedding, document_embeddings)

# Rank documents by similarity
top_k = 10
top_indices = cos_sim[0].argsort()[-top_k:][::-1]

# Display top-k results
print("Top 10 results:")
for idx in top_indices:
    print(f"{documents[idx]} (Score: {cos_sim[0][idx]:.4f})")
