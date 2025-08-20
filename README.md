# T-Retrievability

Topic-Focused Approach to Measure Fair Document Exposure in Information Retrieval

![Localised measure using dense representations of documents for K-means](https://github.com/XuejunChang/T-Retrievability/blob/main/results/aggr_gini_scikit_dense.png)

![Localised measure using tf-idf representations of documents for K-means](https://github.com/XuejunChang/T-Retrievability/blob/main/results/aggr_gini_scikit_tfidf.png)

## Installation
Using git clone into user home directory is recommended. Otherwise, you need to check the directory configurations, such as config.py, plot.ipynb.

```bash
# decompress TREC files. Note that there is enough disk space. 
cd ./T-Retrievability
./decompress.sh
```

## Run

Collection-level Gini evaluation. For example, bm25:

```bash
python ./src/collection_eval.py "bm25_msmarco-passage_dev_100.res"
```

Topical Gini evaluation. For example, bm25:

```bash
python ./src/topical_eval.py "bm25_msmarco-passage_dev_100.res" "clustered_dev_queries_by_5000_scikit_dense.csv"  
```

# Citation

This is the repository of the paper **T-Retrievability: A Topic-Focused Approach to Measure Fair
Document Exposure in Information Retrieval** at CIKM 2025. Please cite:

```bibtex
@inproceedings{DBLP:conf/cikm/t-retrievability,

}
```
