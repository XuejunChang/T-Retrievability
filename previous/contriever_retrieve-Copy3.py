import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()

import os
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
    
    # print(tokenized_inputs)
    # Compute embeddings
    with torch.no_grad():
        model_output = model(**tokenized_inputs)
        token_embeddings = model_output.last_hidden_state 
    
    # Aggregate embeddings (e.g., mean pooling)
    attention_mask = tokenized_inputs["attention_mask"]  # To ignore padding tokens in the aggregation
    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # Apply attention mask
    sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over the sequence length
    sum_mask = attention_mask.sum(dim=1).unsqueeze(-1)  # Count non-padding tokens per sequence
    inputs_embeddings = sum_embeddings / sum_mask  # Mean pooling: divide by token counts
    
    # Display final embeddings (shape: batch_size, hidden_dim)
    # print(inputs_embeddings.shape)  # Example: torch.Size([3, 768])
    # print(inputs_embeddings)  # Example: torch.Size([3, 768])
    return inputs_embeddings

df = pd.DataFrame(dataset.get_corpus_iter(verbose=True))

# if os.exists('document_embeddings.npy'):
#     document_embeddings = np.load("document_embeddings.npy")
# else:
#     print(f'{datetime.datetime.now()} start calc documents embeddings')
#     document_embeddings = calc_embeddings(documents).cpu().numpy()
#     print(f'{datetime.datetime.now()} done')
#     np.save("document_embeddings.npy", document_embeddings)
#     print("saved to document_embeddings.npy")

res_df = pd.DataFrame(None, columns=['qid','docno','rtr_cnt','sim_score'])
def calc_sim(top_k, queries, documents):
    print(f'{datetime.datetime.now()} start calc queries embeddings')
    query_embeddings = calc_embeddings(queries).cpu().numpy()
    print(f'{datetime.datetime.now()} done')

    print(f'{datetime.datetime.now()} start calc documents embeddings')
    document_embeddings = calc_embeddings(documents).cpu().numpy()
    print(f'{datetime.datetime.now()} done')
    # np.save("document_embeddings.npy", document_embeddings)
    # print("saved to document_embeddings.npy")


    # Display the similarity matrix
    # print("Cosine Similarity Matrix:")
    # print(cos_sim_matrix)
    # Retrieve the top-k documents for each query
    global res_df
    for i, query_similarities in pt.tqdm(enumerate(cos_sim_matrix)):
        print(f"query {i}:")
        qid = topics.loc[i,'qid']
        top_indices = query_similarities.argsort()[-top_k:][::-1]
        # top_scores = [query_similarities[idx] for idx in top_indices]
        # print(top_scores)
        for idx in top_indices:
            docno = df.loc[idx,'docno']
            # the doc is retrieved 1 time
            sub_df = pd.DataFrame([[qid, docno,1, query_similarities[idx]]],columns = res_df.columns)
            res_df = pd.concat([res_df,sub_df],ignore_index=True)
            print(res_df)
            # print(f"  Document {idx}: Score = {query_similarities[idx]:.4f}")
    
    # print('start saving to pkl file')
    # res_df.to_pickle('/nfs/datasets/cxj/retrievability-bias/results_contriever.pkl')
    # print('done')

# def retrieve_topk(top_k, q, documents):
#     df['text'].progress_apply(lambda d: retrieve(top_k,q,d))
    
batch_size = 16
for i in range(0,topics.shape[0], batch_size):
    queries = topics[i:i+batch_size]
    
    for j in range(0,df.shape[0],batch_size):
        documents = df[j:j+batch_size]
        calc_sim(queries, documents))


