from more_itertools import chunked
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pyterrier_dr import BiEncoder
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()

class Contriever(BiEncoder):

    def __init__(self, model_name=None, batch_size=16, text_field='text', verbose=False, device=None):
        print(f'Contriever batch size: {batch_size}')
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'device===: {device}')
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.verbose = verbose
        self.batch_size = batch_size

    def encode_queries(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(tqdm(texts), batch_size or self.batch_size):
                inps = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=36)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state
                res = res[:, 4:, :].mean(dim=1) # remove the first 4 tokens (representing [CLS] [ Q ]), and average
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer([f'[CLS] [D] {d}' for d in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state
                res = res[:, 4:, :] # remove the first 4 tokens (representing [CLS] [ D ])
                res = res * inps['attention_mask'][:, 4:].unsqueeze(2) # apply attention mask
                lens = inps['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1)
                lens[lens == 0] = 1 # avoid edge case of div0 errors
                res = res.sum(dim=1) / lens # average based on dim
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    # def transform(self, queries_and_docs):
    #     print(queries_and_docs)
    #     groupby = queries_and_docs.groupby("qid")
    #     results = []
    #     with torch.no_grad():
    #         for qid, group in tqdm(groupby, total=len(groupby), unit="q") if self.verbose else groupby:
    #             query = group["query"].values[0]
    #             query_embedding = self.encode_queries([query])
    #             doc_embeddings = self.encode_docs(group['text'].tolist(),batch_size=self.batch_size)
    #
    #             query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    #             doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    #
    #             similarity_scores = np.dot(doc_norms, query_norm.T)  # Shape: (100, 1)
    #             ranked_indices = np.argsort(-similarity_scores.squeeze())
    #
    #             res_dict = {
    #                 "docids": ranked_indices,
    #                 "scores": similarity_scores[ranked_indices].squeeze()
    #             }
    #
    #             for rank, (docid, score) in enumerate(zip(res_dict["docids"], res_dict["scores"])):
    #                 results.append([qid,query,docid,score,rank])
    #
    #
    #     return pd.DataFrame(results, columns=["qid", "query", "docid", "score", "rank"])


    def __repr__(self):
        return f'Contriever({repr(self.model_name)})'
