
import pandas as pd
import pyterrier as pt

import fair_utils, config
import os,sys
if not pt.java.started():
    pt.java.init()
from tqdm import tqdm
from more_itertools import chunked
import pyterrier_dr
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
batch_size = 32

def encode_trec_res(trec_file):
    trec_df = fair_utils.convert_res2df(f'{config.data_dir}/{trec_file}')
    trec_df = trec_df.merge(config.corpus_df, how="left", on="docno")
    texts = trec_df["text"].tolist()
    vectors = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for batch in tqdm(chunked(texts, batch_size), total=num_batches, desc="Batch encoding"):
        vecs = model.encode(batch)
        vectors.extend(vecs)
    trec_df['doc_vec'] = vectors

    return trec_df

def diversify(trec_df, lbda):
    mmr = pyterrier_dr.MmrScorer(Lambda=lbda, verbose=True)
    result_df = mmr.transform(trec_df)

    result_csv = f'{config.data_dir}/{os.path.splitext(trec_file)[0]}_diver_df_lbda{lbda}.csv'
    print(f'saving into {result_csv}')
    result_df.to_csv(result_csv, index=False)
    print('done')

    result_res = f'{config.data_dir}/{os.path.splitext(trec_file)[0]}_diver_df_lbda{lbda}.res'
    print(f'saving trec res into {result_res}')
    fair_utils.save_trec_res(result_df,result_res,f'{os.path.splitext(trec_file)[0]}_lbda{lbda}')
    print('done')

trec_file = sys.argv[1]
# trec_file = "/mnt/datasets/cxj/exposure-fairness-extend/tctcolbert_msmarco-passage_dev_200.res"
if __name__ == '__main__':
    trec_df = encode_trec_res(trec_file)
    for lbda in [0.0, 0.25,0.5,0.75,0.9,1.0]:
        diversify(trec_df, lbda)

    trec_csv_doc_vec = f'{config.data_dir}/{os.path.splitext(trec_file)[0]}_doc_vec.pkl'
    print(f'saving into {trec_csv_doc_vec}')
    trec_df.to_pickle(trec_csv_doc_vec)
    print('done')

