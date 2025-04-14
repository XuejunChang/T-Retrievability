import pickle as pkl
import pandas as pd
data_dir = '/nfs/datasets/cxj/retrievability-bias/categoried'
# data_dir='/root/retrievability-bias'
import os
import sys

start = int(sys.argv[1])
print(f'start={start}')
end = int(sys.argv[2])
print(f'end={end}')
total_rows=101093


for i in range(start,end+1):
    if i % 200 == 0 or i == total_rows-1:
        srcf = f"{data_dir}/rtrDf_{i}.pkl"
        resf = f"{data_dir}/rtrDf_{i}_scored.csv"
        print(f'converting {srcf} file')
        with open(srcf, "rb") as f:
            file = pkl.load(f)
            
        df = pd.DataFrame(file)
        print('start to_csv')
        df.to_csv(resf,index=False)
        print(f'converted csv saved')

        os.system(f'rm -rf {srcf}')
        print(f'{srcf} deleted.')
        del df

