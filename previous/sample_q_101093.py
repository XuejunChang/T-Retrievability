import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt

if not pt.started():
    pt.init()

import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', False)


dataset_name = 'msmarco-passage'
eval_ds_name = 'dev'
dataset = pt.get_dataset(f'irds:{dataset_name}')
# df = pd.DataFrame(dataset.get_corpus_iter(verbose=True))
# sampled_df = df.sample(n=100000, random_state=41)
# sampled_df.to_csv('sampled_df_100k.csv', index=False)

sampled_df = pd.read_csv('sampled_df_101093.csv', index_col=0).reset_index()
# print(sampled_df.columns)
# add a column querygen

df_gen = pd.DataFrame()
batch_size=1000

from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
from pyterrier_dr import ElectraScorer
scorer = ElectraScorer()
doc2query = Doc2Query(append=False, num_samples=1)
doc2query >> QueryScorer(scorer) >> QueryFilter(t=3.21484375) # t=3.21484375 is the 70th percentile for generated queries on MS MARCO
for batch_start in pt.tqdm(range(0, sampled_df.shape[0], batch_size)):
    batch_df = sampled_df[batch_start:batch_start + batch_size]
    generated_batch = doc2query.transform(batch_df)
    df_gen = pd.concat([df_gen, generated_batch], ignore_index=True)

df_gen.to_csv(f'df_gen_101093.csv', index=False)
