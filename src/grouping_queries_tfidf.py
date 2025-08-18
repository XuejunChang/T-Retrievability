import config
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

topics = config.topics

vectorizer = TfidfVectorizer(stop_words="english")
V = vectorizer.fit_transform(topics["query"])

for nc in config.num_clusters:
    kmeans = KMeans(n_clusters=nc, random_state=42, verbose=1)
    df = topics.copy()
    df["cluster"] = kmeans.fit_predict(V)
    result_file = f'{config.prog_dir}/grouped_queries/clustered_dev_queries_scikit_tfidf_{nc}.csv'
    if os.path.exists(result_file):
        os.remove(result_file)
        print(f'{result_file} removed')

    print(f'saving into {result_file}')
    df.to_csv(result_file, index=False)
    print('done')

