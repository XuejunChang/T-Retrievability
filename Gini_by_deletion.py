import pandas as pd

pd.set_option('display.max_rows', None)
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os

os.chdir('/mnt/primary/exposure-fairness')
import config, fair_utils, convert, evaluate

def eval_deleted_df_gini(res_path,columns,qrels_res, freq_num, cut_off, adj_bottom_num):
    print(f'processing {res_path}')
    df = convert.convert_res2docdf(res_path, columns=columns)

    rscore_df = evaluate.get_every_doc_rscore(res_path)
    grouped = rscore_df.groupby('docid').agg(
        num=('r_score', 'size'),
        sum_r_score=('r_score', 'sum')
    ).reset_index()

    grouped = grouped.sort_values(by=['num'], ascending=False) # docno, num, sum_r_score
    grouped_high_freq = grouped[grouped['num'] >= freq_num] # the docids with higher retrieval frequency will be further processed.

    df_to_del = df[(df['docid'].isin(grouped_high_freq['docid'].values.astype(int))) & (df['rank'] >= (cut_off - adj_bottom_num))]
    df_deleted = df[~ df['docid'].isin(df_to_del['docid'])]
    df_deleted_cut20 = fair_utils.cut_df(df_deleted, cut_off)

    df_deleted_cut20_res = f'{os.path.splitext(res_path)[0]}_fre{freq_num}_del{adj_bottom_num}_cut{cut_off}.res'
    run_name = os.path.splitext(df_deleted_cut20_res)[0].split('/')[-1]
    convert.convert_docdf2_trec(df_deleted_cut20, df_deleted_cut20_res, run_name)

    coll_gini = evaluate.evaluate_coll_gini(df_deleted_cut20_res)
    mertrics = evaluate.evaluate_metrics(qrels_res, df_deleted_cut20_res)

    return coll_gini, mertrics

if __name__ == '__main__':
    retr_num = 100
    freq_num = 10
    cut_off = 20
    adj_bottom_num = 15
    iter_num = 100

    result = []
    for modelname in config.models:
        qrels_res = f'{config.data_dir}/qrels_dev.res'

        # res_path = f'{config.data_dir}/{modelname}_msmarco-passage_dev_100.res'
        res_path = f'{config.data_dir}/{modelname}_msmarco-passage_dev_{retr_num}_mab_iteration_{iter_num}.res'
        columns = ['qid', 'Q0', 'docid', 'rank', 'score', 'run']

        # df = convert.convert_res2docdf(res_path, columns=columns)
        # df_cut20 = fair_utils.cut_off(df, cut_off)
        # df_cut20_res = f'./{os.path.splitext(res_path)[0]}_cut{cut_off}.res'
        # run_name = os.path.splitext(df_cut20_res)[0].split('/')[-1]
        # convert.convert_docdf2_trec(df_cut20, df_cut20_res, run_name)
        # coll_gini = evaluate.evaluate_coll_gini(df_cut20_res)
        # mertrics = evaluate.evaluate_metrics(qrels_res, df_cut20_res)

        coll_gini, mertrics = eval_deleted_df_gini(res_path,columns,qrels_res, freq_num, cut_off, adj_bottom_num)
        print(f'{modelname}, nDCG@10: {mertrics["ndcg_cut_10"]}, map: {mertrics["map"]}')
        result.append([modelname, coll_gini, mertrics["ndcg_cut_10"], mertrics["map"]])

    result_df = pd.DataFrame(result, columns=['modelname', 'coll_gini', 'ndcg_cut_10', 'map'])
    result_df.to_csv(f'{config.data_dir}/result_deleted_df_gini.csv', index=False)






