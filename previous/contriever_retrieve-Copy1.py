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


class Contriever_Model(TransformerBase):
    def __init__(self,
                 modelname="facebook/contriever-msmarco",
                 batch_size=16,
                 text_field='text',
                 verbose=False):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)  # Move model to GPU
        self.text_field = text_field
        self.verbose = verbose
        self.batch_size = batch_size
        print(f'self.batch_size------------{self.batch_size}')

    # Function to encode text into dense vectors
    def encode_texts(self,texts):
        # Tokenize and move input tensors to GPU
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Generate embeddings and move them back to CPU for further processing
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
        return embeddings

    def transform(self, run):
        # queries = df[self.text_field].to_list()
        queries, texts = run['query'], run[self.text_field]

        df["score"] = self.compute_probability(input_text)
        
        # Encode query and documents
        query_embeddings = encode_texts([queries])  # Query embedding
        text_embeddings = encode_texts(texts)  # Document embeddings

        # Compute cosine similarity
        cos_sim = cosine_similarity(query_embeddings, text_embeddings)
                
        # Rank documents by similarity
        top_k = 10
        top_indices = cos_sim[0].argsort()[-top_k:][::-1]
        
        # Display top-k results
        print("Top 10 results:")
        for idx in top_indices:
            print(f"{documents[idx]} (Score: {cos_sim[0][idx]:.4f})")

        print(run)
        return df








