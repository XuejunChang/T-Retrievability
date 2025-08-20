import numpy as np
from tqdm import tqdm
tqdm.pandas()
import fair_utils
import sys
import config

retrieved_res_file = sys.argv[1]
if __name__ == '__main__':
    import convert

    retrieved_res_path = f'{config.data_dir}/{retrieved_res_file}'
    models_coll_gini = {}
    print(f'computing metrics and gini for {retrieved_res_path}')
    df = convert.convert_res2docdf(retrieved_res_path, config.trec_res_columns)
    print('computing r_score of each document')
    df['r_score'] = df['rank'].progress_apply(lambda x: 1.0 / np.log(x + 2))
    summed_doc_rscores = df.groupby("docno")[['r_score']].sum().reset_index()

    print('computing collection-level gini')
    gini = fair_utils.compute_gini(summed_doc_rscores['r_score'].to_dict())

    print('computing collection-level effective metrics')
    metrics = fair_utils.compute_metrics(config.qrels_res_dev, retrieved_res_path)

    print(f"nDCG@10: {metrics['ndcg_cut_10']}, map: {metrics['map']}, gini: f'{gini:.4f}'")



