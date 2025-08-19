# T-Retrievability

Topic-Focused Approach to Measure Fair Document Exposure in Information Retrieval

![Localised measure using dense representations of documents for K-means](https://github.com/XuejunChang/T-Retrievability/blob/main/result_graphs/aggr_gini_scikit_dense.pdf)

![Localised measure using tf-idf representations of documents for K-means](https://github.com/XuejunChang/T-Retrievability/blob/main/result_graphs/aggr_gini_scikit_tfidf.pdf)

## Installation

```bash
# decompress:
cd retrieved_trec_files/ && xz -d *.xz
```

## Run

Collection-level Gini evaluation:

```bash
cd src/ && collection-eval.py
```

Topical Gini evaluation:

```bash
cd src/ && topical_eval.py
```

# Citation

This is the repository of the paper **T-Retrievability: A Topic-Focused Approach to Measure Fair
Document Exposure in Information Retrieval** at CIKM 2025. Please cite:

```bibtex
@inproceedings{DBLP:conf/cikm/t-retrievability,

}
```
