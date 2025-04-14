import pandas as pd
monot5 = pd.read_csv('/nfs/primary/retrievability-bias/results_bm25_monot5_r100.csv',index_col=1)# 100 docs retrieved for each query.
# print(monot5.head(5))
# print('-------------')
# print(monot5.index)

print('start grouping...')
monot5 = monot5.groupby('qid').apply(lambda x: x.sort_values('rank', ascending=True)).groupby('qid').head(10)
monot5.to_csv('/nfs/primary/retrievability-bias/results_bm25_monot5_r10.csv')
print('done')