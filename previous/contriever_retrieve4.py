import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()

import os
import sys
import torch
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
    
def calc_embeddings(inputs):
    # Tokenize the inputs
    tokenized_inputs = tokenizer(inputs,padding=True,truncation=True,return_tensors="pt").to(device)  # Move tokenized inputs to GPU
    
    # Compute embeddings
    with torch.no_grad():
        token_embeddings = model(**tokenized_inputs).last_hidden_state 
    
    # Aggregate embeddings (e.g., mean pooling)
    attention_mask = tokenized_inputs["attention_mask"]  # To ignore padding tokens in the aggregation
    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # Apply attention mask
    sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over the sequence length
    sum_mask = attention_mask.sum(dim=1).unsqueeze(-1)  # Count non-padding tokens per sequence
    inputs_embeddings = sum_embeddings / sum_mask  # Mean pooling: divide by token counts

    del tokenized_inputs
    del token_embeddings
    del attention_mask
    del masked_embeddings
    del sum_embeddings
    del sum_mask
    torch.cuda.empty_cache()
    
    return inputs_embeddings


# for i in range(0,40000, batch_size):
#     queries = topics[i:i+batch_size]['query'].to_list()

#     res = []
#     for j in range(0,df.shape[0],batch_size):
#         documents = df[j:j+batch_size]['text'].to_list()
#         res = res + calc_sim(i, j, queries, documents)

#     res_df = pd.DataFrame(res, columns=['qid','docno','rtr_cnt','sim_score'])
#     print('start saving to csv file')
#     res_df.to_csv(f'/nfs/llm/cxj_models/retr_bias/contriever_msmarco_{batch_q}.csv', index=False)
#     print('done')


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
q_emb_dict = {}
for i in range(d1,d2,batch_size):
    documents = df[i:i+batch_size]['text'].to_list()
    print(f'{datetime.datetime.now()} start calc documents embeddings')
    document_embeddings = calc_embeddings(documents)
    documents_cpu = document_embeddings.cpu().numpy()
    print(f'{datetime.datetime.now()} done')

    res = []
    for j in range(0,2000, batch_size):
        print(f'{datetime.datetime.now()} start ----d:{i} -----q:{j}')
        queries = topics[j:j+batch_size]['query'].to_list()

        if str(j) in q_emb_dict:
            queries_cpu = q_emb_dict[str(j)]
            print(f'{datetime.datetime.now()} loaded queries embeddings from memory')
        else:  
            print(f'start calc queries embeddings')
            query_embeddings = calc_embeddings(queries)
            queries_cpu = query_embeddings.cpu().numpy()
            print(f'{datetime.datetime.now()} done')
            print(f'queries_cpu size: {sys.getsizeof(queries_cpu)/1024}k')
            q_emb_dict[str(j)] = queries_cpu
            
            del query_embeddings
            torch.cuda.empty_cache()

        print(f'start computing cosine similarity')
        cos_sim_matrix = cosine_similarity(queries_cpu, documents_cpu)

        print('start appending results into a res list')
        for q_idx, query_similarities in enumerate(cos_sim_matrix):
            qid = topics.loc[j + q_idx,'qid']
            # print(f"query {qid}:")
            for d_idx, score in enumerate(query_similarities):
                docno = df.loc[i + d_idx,'docno']
                # the doc is retrieved 1 time
                res.append([qid, docno,1, score])

        
    res_df = pd.DataFrame(res, columns=['qid','docno','rtr_cnt','sim_score'])
    print('start saving to csv file')
    res_df.to_csv(f'/nfs/llm/cxj_models/retr_bias/contriever_msmarco_{i}.csv', index=False)
    print('done')
    
    del document_embeddings
    res = []
    res_df = None
    torch.cuda.empty_cache()



