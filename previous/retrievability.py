import pyterrier as pt
if not pt.java.started():
    pt.java.init()
import warnings
warnings.filterwarnings('ignore')
from ir_measures import AP, nDCG, P, R, RR, MRR
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import os
import shutil

result = pd.read_csv('./result.csv')
print(result)