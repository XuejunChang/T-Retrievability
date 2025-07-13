
import pandas as pd
import pyterrier as pt

import fair_utils, config
import os,sys
if not pt.java.started():
    pt.java.init()
from tqdm import tqdm

import pyterrier_dr


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

