import pyterrier as pt

class DocumentFilter(pt.Transformer):
    def __init__(self, qual_signal="prob", threshold=0):
        self.qual_signal = qual_signal
        self.threshold = threshold
        self.count = 0

    def transform(self, df):
        df_pruned = df[df[self.qual_signal] >= self.threshold]
        # print(df_pruned)
        # self.count += 16
        # print('counted==========================:', self.count)
        return df_pruned
    
    # def filter(x,threshold):
    #     for k in x.keys():
    #         if float(k) == threshold:
    #             return x[k]
    #         return ""