import numpy as np

import time
def gini_formal(arr):
    start = time.time()
    arr = np.array(arr)
    if np.all(arr == 0):
        return 0.0
    n = len(arr)
    x_avg = np.mean(arr)
    sum = 0.0
    for i in range(n):
        for j in range(n):
            sum = sum + np.abs(arr[i]-arr[j])
    denom = 2 * n * n * x_avg
    end = time.time()
    print(f'total time: {end-start} s, {(end-start)/60} min')
    return sum/denom

if __name__ == '__main__':
    arr = [1, 3, 2, 5, 4,6,7,8,9,10]

