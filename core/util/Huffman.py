# @yifan
#
__version__ = "2.5.0"
from core.util import myLog
import numpy as np
import huffman

class Huffman():
    def __init__(self):
        self.hist = None
        self.dict = {}
        self.inv_dict = {}
        self.version = '2021.05.20'

    def Hist(self, x, bins=64):
        x = x.reshape(-1).astype('int32')
        res = np.zeros(bins)
        for i in range(bins):
            res[i] = len(x[x == i])
        return res   

    def make_dict(self):
        tmp = []
        for i in range(len(self.hist)):
            tmp.append((str(i), self.hist[i]))
        self.dict = huffman.codebook(tmp)
        self.inv_dict = {v: k for k, v in self.dict.items()}
        
    def fit(self, X, hist=None, p2=True):
        X = X.astype('int16')
        if hist is not None:
            self.hist = hist
        else:
            bins = max(len(np.unique(X)), np.max(X))
            if p2 == True and np.log2(bins) - np.trunc(np.log2(bins)) > 1e-5:
                bins = pow(2, (int)(np.log2(bins))+1)
            self.hist = self.Hist(X, bins=bins)
        self.make_dict()
        return self
        
    def encode(self, X):
        X = X.reshape(-1).astype('int32')
        stream = ''
        for i in range(len(X)):
            try:
                stream += self.dict[str(X[i])]
            except:
                if str(X[i]) != str(-1):
                    myLog('Skip, Key Not Exist! -> '+str(X[i]))
        return stream

    def decode(self, stream, start=0, size=-1):
        if size < 0:
            size = len(stream)
        dX, last, ct = [], start, 0
        for i in range(start, len(stream)):
            if stream[last:i] in self.inv_dict:
                dX.append((int)(self.inv_dict[stream[last:i]]))
                last = i
                ct += 1
            if ct >= size:
                break
        if ct < size:
            dX.append((int)(self.inv_dict[stream[last:]]))
            last = len(stream)
        return np.array(dX), last