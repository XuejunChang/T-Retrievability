import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()

import os
import sys
import torch
import gc
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', False)
import datetime

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
# def calc_embeddings(texts):
#     # Tokenize and move input tensors to GPU
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
#     with torch.no_grad():
#         # Generate embeddings and move them back to CPU for further processing
#         embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
#     return embeddings


def count_tensors(i):
    tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
    print(f"{i} -----:Number of tensors: {len(tensors)}")
    # for obj in tensors:
    #     print(f'obj=={obj.shape}')
    return tensors
    
def calc_embeddings(inputs):
    # Tokenize the inputs
    t1 = count_tensors(1)
    tokenized_inputs = tokenizer(inputs,padding=True,truncation=True,return_tensors="pt").to(device)  # Move tokenized inputs to GPU
    t2 = count_tensors(2)
    # for obj2 in t2:
    #     for obj1 in t1:
    #         if not torch.equal(obj2,obj1):
    #             print(obj2.shape)
    
    # # print(tokenized_inputs)
    # # Compute embeddings
    # with torch.no_grad():
    #     token_embeddings = model(**tokenized_inputs).last_hidden_state 
    # count_tensors(3)
    # # Aggregate embeddings (e.g., mean pooling)
    # attention_mask = tokenized_inputs["attention_mask"]  # To ignore padding tokens in the aggregation
    # count_tensors(4)
    # masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # Apply attention mask
    # count_tensors(5)
    # sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over the sequence length
    # count_tensors(6)
    # sum_mask = attention_mask.sum(dim=1).unsqueeze(-1)  # Count non-padding tokens per sequence
    # count_tensors(7)
    # inputs_embeddings = sum_embeddings / sum_mask  # Mean pooling: divide by token counts
    # count_tensors(8)

    # print(f'before: torch.cuda.memory_reserved: {torch.cuda.memory_reserved()}')  # Check memory before reset
    # print(f'before: torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}')  # Check memory before reset


    # del tokenized_inputs
    # del token_embeddings
    # del attention_mask
    # del masked_embeddings
    # del sum_embeddings
    # del sum_mask
    # torch.cuda.empty_cache()
    

    return inputs_embeddings


def calc_sim(batch_q, batch_d, queries, documents):
    print(f'batch_q-----{batch_q},batch_d-----{batch_d}')
    print(f'{datetime.datetime.now()} start calc queries embeddings')
    query_embeddings = calc_embeddings(queries)
    # queries_cpu = query_embeddings.cpu().numpy()
    # print(f'{datetime.datetime.now()} done')

    # print(f'{datetime.datetime.now()} start calc documents embeddings')
    # document_embeddings = calc_embeddings(documents)
    # documents_cpu = document_embeddings.cpu().numpy()
    # print(f'{datetime.datetime.now()} done')

    # print(f'start computing cosine similarity')
    # cos_sim_matrix = cosine_similarity(queries_cpu, documents_cpu)
    
    # del query_embeddings
    # del document_embeddings
    # torch.cuda.empty_cache()

    # res = []
    # print('start appending results into a res list')
    # for q_idx, query_similarities in enumerate(cos_sim_matrix):
    #     qid = topics.loc[batch_q + q_idx,'qid']
    #     # print(f"query {qid}:")
    #     for d_idx, score in enumerate(query_similarities):
    #         docno = df.loc[batch_d + d_idx,'docno']
    #         # the doc is retrieved 1 time
    #         res.append([qid, docno,1, score])

    return res

# df = pd.DataFrame(dataset.get_corpus_iter(verbose=True))
batch_size = 1000
for i in range(80000, topics.shape[0],batch_size):
    queries = topics[i:i+batch_size]['query'].to_list()

    res_df = pd.DataFrame(None, columns=['qid','docno','rtr_cnt','sim_score'])
    for j in range(0,topics.shape[0],batch_size):
        documents = topics[j:j+batch_size]['query'].to_list()
        res = calc_sim(i, j, queries, documents)
        sub_df = pd.DataFrame(res, columns=res_df.columns)
        res_df = pd.concat([res_df,sub_df],ignore_index=True)

    # print('start saving to csv file')
    # res_df.to_csv(f'/nfs/llm/cxj_models/retr_bias/contriever_msmarco_{batch_q}.csv', index=False)
    # print('done')



