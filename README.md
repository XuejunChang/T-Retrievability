# T-Retrievability:

A Topic-Focused Approach to Measure Fair Document Exposure in Information Retrieval

![Localised measure using dense representations of documents for K-means](https://github.com/XuejunChang/T-Retrievability/blob/main/results/aggr_gini_min_scikit_dense.pnghttps://github.com/XuejunChang/T-Retrievability/blob/main/results/aggr_gini_mean_scikit_dense.png,https://github.com/XuejunChang/T-Retrievability/blob/main/results/aggr_gini_max_scikit_dense.png)

![Localised measure using tf-idf representations of documents for K-means](https://github.com/XuejunChang/T-Retrievability/blob/main/results/aggr_gini_min_scikit_tfidf.png, https://github.com/XuejunChang/T-Retrievability/blob/main/results/aggr_gini_mean_scikit_tfidf.png, https://github.com/XuejunChang/T-Retrievability/blob/main/results/aggr_gini_max_scikit_tfidf.png)

## Installation
Using user's home directory is recommended. Otherwise, you need to check the configurations in config.py and plot.ipynb.

```bash
# Due to package dependencies, you need to create two conda environments.
# Create an environment for Splade.
conda env create -f environment_splade.yml
# Create an environment for other models.
conda env create -f environment.yml 
```

```bash
# decompress TREC files. Note that there is enough disk space. 
cd ./src
./decompress.sh
```

## Run

Collection-level Gini evaluation. For bm25:

```bash
python ./collection_eval.py "bm25_msmarco-passage_dev_100.res"
```

Topical Gini evaluation. For bm25:

```bash
python ./topical_eval.py "bm25_msmarco-passage_dev_100.res" "clustered_dev_queries_by_5000_scikit_dense.csv"  
```

# Citation

This is the repository of the paper **T-Retrievability: A Topic-Focused Approach to Measure Fair
Document Exposure in Information Retrieval** at CIKM 2025. Please cite:

```bibtex
https://doi.org/10.1145/3746252.3760820.

@inproceedings{DBLP:conf/cikm/t-retrievability,

}
```
