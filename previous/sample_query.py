import pyterrier as pt
if not pt.java.started():
    pt.java.init()
import warnings
warnings.filterwarnings('ignore')
from ir_measures import AP, nDCG, P, R, RR, MRR
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import os
import shutil

work_name="retrievability-bias"
root_dir = f'/root/{work_name}'
nfs_dir = f'/nfs/datasets/cxj/{work_name}'
if not os.path.exists(nfs_dir):
    os.makedirs(nfs_dir)

dataset_name = 'msmarco-passage'
eval_ds_name = 'trec-dl-2019'
text_field= 'text'
ranker = "BM25"
retrieve_num = 10
dataset = pt.get_dataset(f'irds:{dataset_name}')
eval_dataset = pt.get_dataset(f'irds:{dataset_name}/{eval_ds_name}')
topics = eval_dataset.get_topics()
qrels = eval_dataset.get_qrels()

# dataset_index = f'/nfs/llm/indices/{dataset_name}'
# index = pt.IndexRef.of(f'{dataset_index}')
index = pt.IndexFactory.of(f'/nfs/llm/indices/{dataset_name}/data.properties')
print(index.getCollectionStatistics())

# if os.path.exists('./df_msmarco.csv'):
#     df = pd.read_csv('./df_msmarco.csv')
# else:
#     df = pd.DataFrame(dataset.get_corpus_iter(verbose=True))
#     df.to_csv('./df_msmarco.csv', index=True)
#
# sample_df = df.sample(n=1000000,random_state=1)
# print(sample_df[0:5])
#
# def is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         pass
#     try:
#         import unicodedata
#         unicodedata.numeric(s)
#         return True
#     except (TypeError, ValueError):
#         pass
#     return False
#
# di = index.getDirectIndex()
# doi = index.getDocumentIndex()
# lex = index.getLexicon()
#
# Q = []
# terms = []
# count = 0
# for docid, _ in sample_df.iterrows():
#     # print(docid)
#     # print(f'terms={terms}')
#     for posting in di.getPostings(doi.getDocumentEntry(docid)):
#         termid = posting.getId()
#         lee = lex.getLexiconEntry(termid)
#         term = lee.getKey()
#         freq = posting.getFrequency()
#         if freq >= 2 and not is_number(term):
#             # print("%s with frequency %d" % (term,freq))
#             terms.append(term)
#             if len(terms) == 3:
#                 Q.append(' '.join(terms))
#                 terms = []
#
#     terms = []
#     # count +=1
#     # if count ==8:
#     #     break
# df_Q = pd.DataFrame({'qid':range(0,len(Q)),'query':Q})

# df_Q = pd.read_csv('./df_Q.csv')
# bm25 = pt.terrier.Retriever(index, wmodel="BM25") % 10
# pipe = bm25 >> pt.text.get_text(dataset,'text')
# sample_Q = df_Q.sample(n=100000,random_state=1)
# print('start transforming...')
# result_all = pipe.transform(sample_Q)
# result_all.to_csv('./result_all.csv')
# g = result_all.groupby(['docid'])['docid'].count()
# df_g = pd.DataFrame(g)
# df_g.index.name='inx'
# df_g.sort_values(by='docid')
# print(df_g)

#########################################################
# result_all = pd.read_csv('./result_all.csv')
# g = result_all.groupby(['docid'])['docid'].count()
# df_g = pd.DataFrame(g)
# df_g.index.name='inx'
# df_sorted = df_g.sort_values(by='docid')
# print(df_sorted)
# df_sorted.to_csv('./df_sorted.csv')

##########################
df_Q = pd.read_csv('./df_Q.csv')
tf_idf = pt.terrier.Retriever(index, wmodel="TF_IDF")
pipe = tf_idf >> pt.text.get_text(dataset,'text')
sample_Q = df_Q.sample(n=100000,random_state=1)
print('start transforming...')
result_all = pipe.transform(sample_Q)
g = result_all.groupby(['docid'])['docid'].count()
df_g = pd.DataFrame(g)
df_g.index.name='inx'
df_sorted_tfidf = df_g.sort_values(by='docid')
print(df_sorted_tfidf)
df_sorted_tfidf.to_csv('./df_sorted_tfidf.csv')