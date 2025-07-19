import math
import os
import subprocess
import pandas as pd
import config
import glob
import argparse
"""
map_values: a map instance
"""
def compute_gini(map_values):
    values = sorted(map_values.values())
    n = len(values)
    if n == 0:
        return 0.0

    total = sum(values)
    if total == 0:
        return 0.0

    gini_numerator = sum((2 * (i + 1) - n - 1) * val for i, val in enumerate(values))
    return gini_numerator / (n * total)

"""
Note: the cluster values have been sorted in ascending order.
"""
def build_log_reciprocal_rank_map(filename, modelname, granu, km):
    base_key = f'{modelname}_granu_{granu}_{km}'
    rr_map = None
    gini_map = {}
    last_cluster = -1
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split() # columns=["qid", "Q0", "docid", "rank", "score", "run", "cluster"]
            if len(parts) < 4:
                continue  # skip invalid lines

            try:
                cluster = int(parts[6])
            except Exception:
                continue

            docid = parts[2]
            try:
                rank = int(parts[3]) + 1
                if rank <= 0:
                    continue  # avoid invalid rank
                rr = 1.0 / math.log(1 + rank)

                if last_cluster == -1:
                    rr_map = {}
                elif cluster != last_cluster:
                    cluster_key = f'{base_key}_{last_cluster}'
                    gini_map[cluster_key] = compute_gini(rr_map)
                    rr_map = {}

                rr_map[docid] = rr_map.get(docid, 0.0) + rr
                last_cluster = cluster

            except (ValueError, ZeroDivisionError, OverflowError):
                continue  # skip malformed lines
            except Exception as e:
                print(f"Error occurred: {e}")

    values = gini_map.values()
    average = sum(values) / len(values)
    minimum = min(values)
    maximum = max(values)
    return base_key, f'{minimum:.4f}', f'{average:.4f}', f'{maximum:.4f}'
    
def cal_metrics(qrels_path, docs_path):
    # ensure that cp /mnt/primary/exposure-fairness/trec_eval /usr/local/bin/
    # all_metrics = [
    #     "map", "set_map", "set_P", "set_recall", "set_F", "Rprec", "bpref", "recip_rank",
    #     "ndcg", "ndcg_cut.5", "ndcg_cut.10", "ndcg_cut.20",
    #     "P.5", "P.10", "P.20", "P",
    #     "recall.5", "recall.10", "recall.20", "recall",
    #     "iprec_at_recall.0.0", "iprec_at_recall.0.1", "iprec_at_recall.0.2",
    #     "iprec_at_recall.0.3", "iprec_at_recall.0.4", "iprec_at_recall.0.5",
    #     "iprec_at_recall.0.6", "iprec_at_recall.0.7", "iprec_at_recall.0.8",
    #     "iprec_at_recall.0.9", "iprec_at_recall.1.0",
    #     "F.5", "F.10", "F.20", "F", "relstring"
    # ]

    metrics = ["ndcg_cut.10", "map", "recip_rank", "P.10"]
    args = []
    for metric in metrics:
        args.append('-m')
        args.append(metric)
    cmd = ["trec_eval"] + args + [qrels_path, docs_path]
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    metric_dict = {}
    for line in result.stdout.splitlines():
        arr = line.split()
        metric_dict[arr[0]] = float(arr[-1])

    print(metric_dict)
    return metric_dict

# model_name = sys.argv[1]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Gini coefficient.")
    parser.add_argument("filename", help="Path to the input space-separated file")
    args = parser.parse_args()

    rr_map = build_log_reciprocal_rank_map(args.filename)
    gini = compute_gini(rr_map)
    print(f"Gini coefficient: {gini:.6f}")

    qrels_path = '/nfs/datasets/cxj/exposure-fairness/v1/qrels_dev.res'
    docs_path = '/nfs/datasets/cxj/exposure-fairness/v1/bm25_tctcolbert_100.res'
    qrels_path = f'{config.data_dir}/qrels_dev.res'
    docs_path = f'{config.data_dir}/bm25_100.res'

    # result = cal_metrics(qrels_path, docs_path)
    # batch_convert_df2trec(f'{config.prog_dir}/grouped_queries')