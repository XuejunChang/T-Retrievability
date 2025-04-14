import pyterrier as pt
if not pt.started():
    pt.init()

import os
import DocumentFilter
import pyt_splade

def create_index(inx, threshold, modelname, rankername, cache_file, nfs_dir, qual_signal=None):
    index_file = f"{nfs_dir}/splade/{modelname}-{rankername}-nostemmer-nostopwords-index-{str(threshold)}"

    print(f"indexing into {index_file}")
    splade = pyt_splade.Splade()
    pipe = DocumentFilter(qual_signal=qual_signal, threshold=threshold) >> splade.doc_encoder() >> pt.IterDictIndexer(index_file, stemmer=pt.TerrierStemmer.none, stopwords=pt.TerrierStemmer.none, pretokenised=True, verbose=True)
    pipe.index(pt.tqdm(cache_file.get_corpus_iter()))

    print('indexing done')

def evaluate_experiment(inx, threshold, modelname, dataset, rankername, topics, topics_ins, retrieve_num, nfs_dir):
    index_file = f"{nfs_dir}/{rankername}/{modelname}-{rankername}-nostemmer-nostopwords-index-{str(threshold)}"
    if not os.path.exists(index_file):
        print(f"Index file {index_file} does not exist")
        return

    splade = pyt_splade.Splade()
    retriever = splade.query_encoder() >> pt.terrier.Retriever(index_file, wmodel='Tf', verbose=True)
    retriever = retriever % retrieve_num

    csv = f'{nfs_dir}/{rankername}/df_{rankername}_{topics}_{inx * 30}.csv'
    if os.path.exists(csv):
        print(f"File {csv} already exists")
        # df = pd.read_csv(csv,index_col=1).reset_index()
        return
    else:
        print(f'start tramsforming {rankername} on {topics} at {inx * 30}%')
        df = retriever.transform(topics_ins)
        print(f'df of {rankername} {topics} columns {df.columns.tolist()}')
        df = df[['qid','docid','docno','score','rank']]
        print(f'opt in in df of {rankername} {topics} shape {df.shape}')
        df.to_csv(csv,index=False)
        print(f'save {rankername} {topics} with shape {df.shape} into {csv}')


