import math
import subprocess

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

def build_log_reciprocal_rank_map(filename):
    rr_map = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # skip invalid lines
            docid = parts[2]
            try:
                rank = int(parts[3]) + 1
                if rank <= 0:
                    continue  # avoid invalid rank
                rr = 1.0 / math.log(1 + rank)
                rr_map[docid] = rr_map.get(docid, 0.0) + rr
            except (ValueError, ZeroDivisionError, OverflowError):
                continue  # skip malformed lines
    return rr_map

"""
Note: the cluster values have been sorted in ascending order.
"""
def compute_topic_gini(filename, modelname, granu, km):
    base_key = f'{modelname}_{km}_{granu}'
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

def compute_metrics(qrels_path, docs_path):
    metrics = ["ndcg_cut.10", "map"]
    args = []
    for metric in metrics:
        args.append('-m')
        args.append(metric)
    cmd = ["trec_eval"] + args + [qrels_path, docs_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    metric_dict = {}
    for line in result.stdout.splitlines():
        arr = line.split()
        metric_dict[arr[0]] = float(arr[-1])

    sorted_dict = {k: metric_dict[k] for k in sorted(metric_dict, reverse=True)}
    return sorted_dict
