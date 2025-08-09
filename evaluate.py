import pandas as pd
import pyterrier as pt
if not pt.java.started():
    pt.java.init()
from tqdm import tqdm
import pyterrier_dr
import config, fair_utils, convert
import os, sys, numpy as np
from more_itertools import chunked
from sentence_transformers import SentenceTransformer
import argparse

def encode_trec_res(res_file_path, batch_size):
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

def get_every_doc_rscore(res_file_path,columns=None):
    rscore_csv_path = os.path.splitext(res_file_path)[0] + "_rscore.csv"
    if os.path.exists(rscore_csv_path):
        print(f'Exists {rscore_csv_path}')
        # df = pd.read_csv(rscore_csv_path, index_col=0).reset_index()
        # print('loaded')
        os.remove(rscore_csv_path)
        print('removed')

    df  = convert.convert_res2docdf(res_file_path, columns=columns)
    print('calc r_score of each document of the retrieved docs ...')
    df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))

    print(f'saving into {rscore_csv_path}')
    df.to_csv(rscore_csv_path, index=False)
    print(f'done')

    rscore_res_path = f'{os.path.splitext(rscore_csv_path)[0]}.res'
    convert.convert_dfall2trec(df, rscore_res_path)
    print(f'done')

    return df

def evaluate_coll_gini(res_file_path, columns=None):
    df = get_every_doc_rscore(res_file_path, columns=columns)
    summed_doc_rscores = df.groupby("docid")[['r_score']].sum().reset_index()
    print('calc coll_gini')
    coll_gini = fair_utils.compute_gini(summed_doc_rscores['r_score'].to_dict())
    return coll_gini

def evaluate_metrics(qrels_res, res_file_path):
    return fair_utils.cal_metrics(qrels_res, res_file_path)

def evaluate_res_gini_metrics(res_file_path, cut_off, columns=None):
    print(f'cutting {res_file_path} by top {cut_off}')
    df = convert.convert_res2docdf(res_file_path, columns=columns)

    df_cut20 = fair_utils.cut_df(df, cut_off)
    df_cut20_res = f'{os.path.splitext(res_file_path)[0]}_cut{cut_off}.res'
    run_name = os.path.splitext(df_cut20_res)[0].split('/')[-1]
    convert.convert_docdf2_trec(df_cut20, df_cut20_res, run_name)

    gini = evaluate_coll_gini(df_cut20_res,columns=columns)
    # mertrics = evaluate_metrics(qrels_res, df_cut20_res)
    return gini, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Accept multiple methods and models")
    parser.add_argument('--run', nargs='+', help='Which function to be ran')
    args = parser.parse_args()
    run = args.run

    retr_num = 100
    cut_off = 20
    qrels_res = f'{config.data_dir}/qrels_dev.res'

    if 'diversify' in run:
        batch_size = 32
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        # model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
        for modelname in config.models:
            res_file = f'{modelname}_msmarco-passage_dev_{retr_num}.res'
            res_file_path = f'{config.data_dir}/{res_file}'
            trec_df = encode_trec_res(res_file_path, batch_size)
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
                fair_utils.cut_df(res_file_path, cut_off)

    if 'cut_df' in run:
        for modelname in config.models:
            # res_file_path = f'{config.data_dir}/{modelname}_msmarco-passage_dev_100.res'
            res_file_path = f'{config.data_dir}/{modelname}_msmarco-passage_dev_{retr_num}_mab_iteration_100.res'
            columns = ['qid', 'Q0', 'docid', 'rank', 'score', 'run']

            g100, metrics100 = evaluate_res_gini_metrics(res_file_path, cut_off, columns=columns)

            res_file_path = f'/mnt/datasets/cxj/exposure-fairness-extend/{modelname}_msmarco-passage_dev_200.res'
            g200, metrics200 = evaluate_res_gini_metrics(res_file_path, cut_off, columns=columns)
            # print(f'{modelname},g100:{g100:.4f}, g200:{g200:.4f}, metrics100: {metrics100}, metrics200: {metrics200}')
            print(f'{modelname},g100:{g100:.4f}, g200:{g200:.4f}')

    if 'evaluate_coll_lambda' in run:
        res_list = []
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






