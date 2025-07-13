import warnings
warnings.filterwarnings('ignore')
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os, sys, pandas as pd
import fair_utils
import config

# order of args: [version] retrieve
version = sys.argv[1]
data_dir = f'{config.data_dir}/{version}'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

modelname = sys.argv[2]
run = sys.argv[3:]

if __name__ == '__main__':
    
    if 'eff_metrics' in run:
        qrels_res_path = f'{data_dir}/qrels_dev.res'
        qrels_path = fair_utils.get_trec_qrels(config.qrels, qrels_res_path)
        docs_path = f'{data_dir}/{modelname}_{config.dataset_name}_{config.topics_name}_{config.retrieve_num}.res'
        metrics_dict = fair_utils.cal_metrics(qrels_path, docs_path)
        print(metrics_dict)
