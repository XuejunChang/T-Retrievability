import pandas as pd
import pyterrier as pt
if not pt.java.started():
    pt.java.init()
from tqdm import tqdm
import pyterrier_dr
import config, fair_utils, gini
import os, sys, numpy as np
from more_itertools import chunked
from sentence_transformers import SentenceTransformer

def encode_trec_res(res_file_path):
    trec_df = fair_utils.convert_res2df(res_file_path)
    trec_df = trec_df.merge(config.corpus_df, how="left", on="docno")
    texts = trec_df["text"].tolist()
    vectors = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for batch in tqdm(chunked(texts, batch_size), total=num_batches, desc="Batch encoding"):
        vecs = model.encode(batch)
        vectors.extend(vecs)
    trec_df['doc_vec'] = vectors

    return trec_df

def diversify(trec_df, res_file_path, lbda):
    mmr = pyterrier_dr.MmrScorer(Lambda=lbda, verbose=True)
    result_df = mmr.transform(trec_df)

    result_csv_path = f'{os.path.splitext(res_file_path)[0]}_diver_df_lbda{lbda}.csv'
    print(f'saving into {result_csv_path}')
    result_df.to_csv(result_csv_path, index=False)
    print('done')

    result_res_path = f'{os.path.splitext(result_csv_path)[0]}.res'
    print(f'saving trec res into {result_res_path}')
    run_name = os.path.splitext(result_res_path)[0].split('/')[-1]
    fair_utils.save_trec_res(result_df,result_res_path,run_name)
    print('done')

def get_every_doc_rscore(res_file_path):
    rscore_path = os.path.splitext(res_file_path)[0] + "_rscore.csv"
    if os.path.exists(rscore_path):
        print(f'loading {rscore_path}')
        df = pd.read_csv(rscore_path, index_col=0).reset_index()
        print('loaded')
    else:
        df  = fair_utils.convert_res2df(res_file_path)
        print('calc r_score of each document of the retrieved docs ...')
        df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))

        print(f'saving into {rscore_path}')
        df.to_csv(rscore_path, index=False)
        print(f'done')

    return df

def cut_df(res_file_path, cut_off):
    diver_df = fair_utils.convert_res2df(res_file_path)
    cutted_df = diver_df.groupby("qid", group_keys=False).head(cut_off).reset_index()

    cutted_csv_path = f'{os.path.splitext(res_file_path)[0]}_cut{cut_off}.csv'
    print(f'saving into {cutted_csv_path}')
    cutted_df.to_csv(cutted_csv_path, index=False)
    print('done')

    cutted_res_path = f'{os.path.splitext(cutted_csv_path)[0]}.res'
    run_name = os.path.splitext(cutted_res_path)[0].split('/')[-1]
    print(f'saving trec res into {cutted_res_path}')
    fair_utils.save_trec_res(cutted_df, cutted_res_path, run_name)
    print('done')

def evaluate_coll_gini(res_file_path):
    df = get_every_doc_rscore(res_file_path)
    summed_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()
    print('calc coll_gini')
    coll_gini = gini.compute_gini(summed_doc_rscores['r_score'].to_dict())
    print(f'coll_gini: {coll_gini:.4f}')

    return coll_gini

def evaluate_metrics(qrels_res, res_file_path):
    return fair_utils.cal_metrics(qrels_res, res_file_path)

cut_off = 20
batch_size = 32

retr_num = sys.argv[1]
run = sys.argv[2:]

if __name__ == '__main__':
    if 'diversify' in run:
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        for modelname in config.models:
            res_file = f'{modelname}_msmarco-passage_dev_{retr_num}.res'
            res_file_path = f'{config.data_dir}/{res_file}'
            trec_df = encode_trec_res(res_file_path)
            for lbda in [0.0, 0.25, 0.5, 0.75, 0.9]:
                diversify(trec_df, res_file_path, lbda)

            trec_csv_doc_vec = f'{os.path.splitext(res_file_path)[0]}_doc_vec.pkl'
            print(f'saving into {trec_csv_doc_vec}')
            trec_df.to_pickle(trec_csv_doc_vec)
            print('saved')

    if 'cut_diversified' in run:
        for modelname in config.models:
            res_file = f'{modelname}_msmarco-passage_dev_{retr_num}.res'
            for lbda in [0.0, 0.25, 0.5, 0.75, 0.9]:
                res_file = f'{os.path.splitext(res_file)[0]}_lbda{lbda}.res'
                res_file_path = f'{config.data_dir}/{res_file}'
                cut_df(res_file_path, cut_off)

    if 'cut_df' in run:
        for modelname in config.models:
            res_file = f'{modelname}_msmarco-passage_dev_{retr_num}.res'
            print(f'cutting {res_file} into {cut_off}')
            res_file_path = f'{config.data_dir}/{res_file}'
            cut_df(res_file_path, cut_off)

    if 'evaluate_coll_lambda' in run:
        res_list = []
        qrels_res = f'{config.data_dir}/qrels_dev.res'
        for modelname in config.models:
            res_file = f'{modelname}_msmarco-passage_dev_200.res'
            for lbda in [0.0, 0.25, 0.5, 0.75, 0.9]:
                res_file = f'{os.path.splitext(res_file)[0]}_diver_df_lbda{lbda}_cut{cut_off}.res'
                res_file_path = f'{config.data_dir}/{res_file}'
                coll_gini = evaluate_coll_gini(res_file_path)

                result = evaluate_metrics(qrels_res, res_file_path)
                res_list.append([modelname, lbda, cut_off, f"{coll_gini:.4f}", f"{result['ndcg_cut_10']:.4f}",f"{result['map']:.4f}"])

        result_df = pd.DataFrame(res_list, columns=['modelname', 'Lambda', 'cut_off', 'gini', 'nDCG@10', 'map'])
        result_csv = f'{config.data_dir}/all_models_gini_ndcg_lbda{lbda}_cut{cut_off}.csv'
        result_df.to_csv(result_csv, index=False)

    if 'evaluate_coll_cutted_res' in run:
        res_list = []
        qrels_res = f'{config.data_dir}/qrels_dev.res'
        for modelname in config.models:
            res_file = f'{modelname}_msmarco-passage_dev_{retr_num}.res'
            print(f'evaluating {res_file}')
            res_file_path = f'{config.data_dir}/{res_file}'
            cutted_csv_path = f'{os.path.splitext(res_file_path)[0]}_cut{cut_off}.res'
            coll_gini = evaluate_coll_gini(cutted_csv_path)

            result = evaluate_metrics(qrels_res, cutted_csv_path)
            modelname = os.path.splitext(res_file)[0].split('_')[0]
            res_list.append([modelname, cut_off, f"{coll_gini:.4f}", f"{result['ndcg_cut_10']:.4f}",f"{result['map']:.4f}"])

        result_df = pd.DataFrame(res_list, columns=['modelname', 'cut_off', 'gini', 'nDCG@10', 'map'])
        result_csv = f'{config.data_dir}/all_models_gini_ndcg_map_cut{cut_off}_from_{retr_num}.csv'
        print(f'saving into {result_csv}')
        result_df.to_csv(result_csv, index=False)
        print('saved')






