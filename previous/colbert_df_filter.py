import glob
import pandas as pd

for file in glob.glob('/nfs/datasets/cxj/retrievability-bias/categoried/*.csv'):
    name = file.split('/')[-1]
    df = pd.read_csv(file)
    df = df[['qid','docid','rank']]
    print(f'starting {name}')
    df.to_csv(f'/nfs/datasets/cxj/retrievability-bias/categoried/filtered/{name}',index=False)
    print('done')
    del df