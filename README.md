# T-Retrievability

Topic-Focused Approach to Measure Fair Document Exposure in Information Retrieval

![Localised measure using dense representations of documents for K-means](https://github.com/XuejunChang/T-Retrievability/results/aggr_gini_scikit_dense.png)

![Localised measure using tf-idf representations of documents for K-means](https://github.com/XuejunChang/T-Retrievability/results/aggr_gini_scikit_tfidf.png)

## Installation

```bash
# decompress TREC files. Note that there is enough disk space.
./decompress.sh
```

## Run

Collection-level Gini evaluation. For example, bm25:

```bash
cd src/ && collection-eval.py "../retrieved_trec_files/bm25_monot5_msmarco-passage_dev_100.res"
```

Topical Gini evaluation. For example, bm25:

```bash
cd src/ && topical_eval.py "../retrieved_trec_files/bm25_monot5_msmarco-passage_dev_100.res" "../grouped_queries/clustered_dev_queries_by_5000_scikit_tfidf.csv"  
```

# Citation

This is the repository of the paper **T-Retrievability: A Topic-Focused Approach to Measure Fair
Document Exposure in Information Retrieval** at CIKM 2025. Please cite:

```bibtex
@inproceedings{DBLP:conf/cikm/t-retrievability,

}
```
