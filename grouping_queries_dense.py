import config
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer("all-MiniLM-L6-v2")
topics = config.topics
query_embeddings = model.encode(topics['query'].tolist(), show_progress_bar=True)
num_clusters = config.num_clusters
for nc in config.num_clusters:
    kmeans = KMeans(n_clusters=nc, random_state=42, verbose=1)
    df = topics.copy()
    df["cluster"] = kmeans.fit_predict(query_embeddings)

    result_file = f'{config.prog_dir}/grouped_queries/clustered_dev_queries_by_{nc}_scikit_dense.csv'
    if os.path.exists(result_file):
        os.remove(result_file)
        print(f'{result_file} removed')

    print(f'saving into {result_file}')
    df.to_csv(result_file, index=False)
    print('done')
