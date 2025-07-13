import pandas as pd
import pyterrier as pt
if not pt.java.started():
    pt.java.init()
from tqdm import tqdm
import pyterrier_dr
import config, fair_utils, gini
import os, sys, numpy as np


def get_every_doc_rscore(res_file):
    rscore_csv = os.path.splitext(res_file)[0] + "_rscore.csv"
    rscore_path = f'{config.data_dir}/{rscore_csv}'
    if os.path.exists(rscore_path):
        print(f'loading {rscore_path}')
        df = pd.read_csv(rscore_path, index_col=0).reset_index()
        print('loaded')
    else:
        df  = fair_utils.convert_res2df(res_file)
        print('calc r_score of each document of the retrieved docs ...')
        df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))

        print(f'saving into {rscore_path}')
        df.to_csv(rscore_path, index=False)
        print(f'done')

    return df

def cut_diversified_df(res_file, lbda, cut_off):
        res_file_lbda = f'{config.data_dir}/{os.path.splitext(res_file)[0]}_lbda{lbda}.res'
        diver_df = fair_utils.convert_res2df(res_file_lbda)
        cutted_df = diver_df.groupby("qid", group_keys=False).head(cut_off).reset_index()

        cutted_csv = f'{os.path.splitext(res_file_lbda)[0]}_cut{cut_off}.csv'
        print(f'saving into {cutted_csv}')
        cutted_df.to_csv(cutted_csv, index=False)
        print('done')

        cutted_res = f'{os.path.splitext(res_file_lbda)[0]}_cut{cut_off}.res'
        run_name = os.path.splitext(res_file_lbda)[0].split('/')[-1] + '_cut{cut_off}'
        print(f'saving trec res into {cutted_res}')
        fair_utils.save_trec_res(cutted_df, cutted_res, run_name)
        print('done')

def cut_raw_df(res_file, cut_off):
    res_file_path = f'{config.data_dir}/{res_file}'
    diver_df = fair_utils.convert_res2df(res_file_path)
    cutted_df = diver_df.groupby("qid", group_keys=False).head(cut_off).reset_index()

    cutted_csv = f'{os.path.splitext(res_file_path)[0]}_cut{cut_off}.csv'
    print(f'saving into {cutted_csv}')
    cutted_df.to_csv(cutted_csv, index=False)
    print('done')

    cutted_res = f'{os.path.splitext(res_file_path)[0]}_cut{cut_off}.res'
    run_name = os.path.splitext(res_file_path)[0].split('/')[-1] + '_cut{cut_off}'
    print(f'saving trec res into {cutted_res}')
    fair_utils.save_trec_res(cutted_df, cutted_res, run_name)
    print('done')

def evaluate_collection(res_file):
    df = get_every_doc_rscore(res_file)
    summed_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()
    print('calc coll_gini')
    coll_gini = gini.compute_gini(summed_doc_rscores['r_score'].to_dict())
    print(f'coll_gini: {coll_gini}')

    return coll_gini

cut_off = 20
run = sys.argv[1]
if len(sys.argv) > 2:
    res_file = sys.argv[2] # e.g., bm25_msmarco-passage_dev_200.res

if __name__ == '__main__':
    if run == 'cut_diversified':
        for lbda in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
            cut_diversified_df(res_file, lbda, cut_off)
    if run == 'cut_raw':
        cut_raw_df(res_file, cut_off)

    if run == 'evaluate_collection':
        res_list = []
        qrels_res = 'qrels_dev.res'
        for modelname in config.models:
            for lbda in [0.0, 0.25, 0.5, 0.75, 0.9]:
                res_file = f'{modelname}_msmarco-passage_dev_200.res'
                res_file_lbda = f'{os.path.splitext(res_file)[0]}_diver_df_lbda{lbda}_cut{cut_off}.res'
                coll_gini = evaluate_collection(res_file_lbda)

                result = fair_utils.cal_metrics(qrels_res, res_file_lbda)
                res_list.append([modelname, lbda, cut_off, f"{coll_gini:.4f}", f"{result['ndcg_cut_10']:.4f}",f"{result['map']:.4f}"])

        result_df = pd.DataFrame(res_list, columns=['modelname', 'Lambda', 'cut_off', 'gini', 'nDCG@10', 'map'])
        result_csv = f'{config.data_dir}/all_models_gini_ndcg_lbda{lbda}_cut{cut_off}.csv'
        result_df.to_csv(result_csv, index=False)





