from core.cwSaab import cwSaab
from core.util.myKMeans import myKMeans
from core.util.Huffman import Huffman
import numpy as np
from core.util import Time, myLog, Shrink
from core.util.ac import HierarchyCABAC
from core.util.evaluate import MSE
from core.util.ReSample import *
myLog('<FRAMEWORK> rdVQ 2022.12.09')
Lagrange_multip = 300000/1024**2

def toSpatial(cwSaab, iR, level, S,tX):
    for i in range(level, -1, -1):
        if i > 0:
            s = S[i-1]
            iR = cwSaab.inverse_transform_one(iR, tX[i-1], i)
        else:
            iR = cwSaab.inverse_transform_one(iR, None, i)
    return iR

class VQ:
    def __init__(self, n_clusters_list, win_list, n_dim_list, enable_skip={}, transform_split=0):
        self.n_clusters_list = n_clusters_list
        self.win_list = win_list
        self.n_dim_list = n_dim_list
        self.cwSaab = cwSaab(win=win_list, TH=-1, transform_split=transform_split)
        self.shape = {}
        self.myKMeans = {}
        self.Huffman = {}
        self.buffer = {}
        self.acc_bpp = 0

    def to_spatial(self,iR, tX, level):
        for i in range(level, -1, -1):
            if i > 0:
                iR = self.cwSaab.inverse_transform_one(iR, np.zeros_like(tX[i-1]), i)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, i)
        return iR

    # find the optimal threshold
    def RD_search_th(self, myhash, dmse, mse, omse, pidx, label, S, gidx, h0, ii, isfit):
        min_cost, th = 1000000000, -1
        for skip_TH in range(0, 100000, 200):
            idx = (dmse > skip_TH).reshape(-1)
            if isfit == True:
                st0=''
                l = 2.75*np.sum(idx)/len(idx)+0.00276
                l = int(l*len(idx))
                for i in range(l+1):
                    st0+='0'
            else:
                st0 = HierarchyCABAC().encode(pidx, idx.reshape(S), 1) 
            st2 = h0.encode(gidx.reshape(-1)[idx.reshape(-1)==True])
            st1 = ''
            for i in range(len(ii)):
                h1 = self.Huffman[myhash+'_'+str(i)][ii[i]]
                if h1 is not None:
                    st1 += h1.encode(label.reshape(-1)[idx.reshape(-1)])
            r = len(st0+st1+st2) / S[0]# / 1024**2
            d = np.zeros_like(mse)
            d[idx.reshape(-1)] += mse[idx.reshape(-1)]
            d[idx.reshape(-1)==False] += omse[idx.reshape(-1)==False]
            d = np.mean(d)
            cost = d + Lagrange_multip * r
            if min_cost > cost:
                min_cost = cost
                th = skip_TH
        return th, [min_cost, r, d]

    # compute the rd cost for given iX
    def RD(self, tX, X, iX, label, gidx, level, pos, pidx=None, ii=[], isfit=False):
        myhash = 'L'+str(level)+'-P'+str(pos)
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        siX = np.zeros_like(X)
        siX += iX
        sX, siX = self.to_spatial(X, tX, level), self.to_spatial(siX, tX, level)
        acc_win = 1
        for i in range(0, level+1):
            acc_win *= self.win_list[i]
        sX, siX = Shrink(sX, acc_win), Shrink(siX, acc_win)
        sX, siX = sX.reshape(-1, sX.shape[-1]), siX.reshape(-1, siX.shape[-1])
        dmse = (np.mean(np.square(sX), axis=1)-np.mean(np.square(sX-siX),axis=1))
        mse = (np.mean(np.square(sX-siX),axis=1))
        omse =  np.mean(np.square(sX), axis=1)
        th, cost = self.RD_search_th(myhash, dmse, mse, omse, pidx, label, S, gidx, self.Huffman[myhash+'_gidx'], ii, isfit)
        idx = (dmse > th).reshape(-1)
        return th, cost, idx

    # for each content select suitable codebook
    def RD_search_km(self, tX, X, gidx, level, pos, pidx, isfit=False):
        myhash = 'L'+str(level)+'-P'+str(pos)
        label = np.zeros_like(gidx).reshape(-1)
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        iX = np.zeros_like(X).reshape(-1, X.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        TH, min_cost, skip_idx, tiX = 0, [1000000], None, None
        for i0 in range(len(self.myKMeans[myhash+'_0'])):
            for i1 in range(len(self.myKMeans[myhash+'_1'])):
                for i2 in range(len(self.myKMeans[myhash+'_2'])):
                    for i3 in range(len(self.myKMeans[myhash+'_3'])):
                        km = [self.myKMeans[myhash+'_0'][i0],
                            self.myKMeans[myhash+'_1'][i1],
                            self.myKMeans[myhash+'_2'][i2],
                            self.myKMeans[myhash+'_3'][i3]]
                        for i in  range(4):
                            label[gidx.reshape(-1)==i] = km[i].predict(X[gidx.reshape(-1)==i,:self.n_dim_list[level][pos]]).reshape(-1)
                            iX[gidx.reshape(-1)==i,:self.n_dim_list[level][pos]] = km[i].inverse_predict(label[gidx.reshape(-1)==i].reshape(-1,1))
                        th, cost, idx = self.RD(tX, X.reshape(S), iX.reshape(S), label, gidx, level, pos, pidx, [i0,i1,i2,i3], isfit)
                        if cost[0] < min_cost[0]:
                            TH = th
                            min_cost = cost
                            skip_idx = idx
                            tiX = iX
        myLog('<INFO> RD_cost=%8.4f r=%f d=%4.5f Skip_TH=%d'%(min_cost[0], min_cost[1], min_cost[2], th))
        tiX = tiX.reshape(-1, tiX.shape[-1])
        tiX[skip_idx ==  False] *= 0 
        myLog('<BITSTREAM> bpp=%f'%min_cost[1])
        self.buffer[myhash+'_idx'] = skip_idx
        self.buffer[myhash+'_label'] = label
        self.buffer[myhash+'_gidx'] = gidx
        self.buffer[myhash+'_th'] = TH
        return tiX.reshape(S)
        
    @Time
    def fit_one_level_one_pos(self, X, tX, level, pos, gidx):
        myhash = 'L'+str(level)+'-P'+str(pos)
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        S = X.shape
        iX = np.zeros_like(X).reshape(-1, X.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        for i in range(4):
            ii = gidx.reshape(-1) == i
            nc = self.n_clusters_list[level][pos]
            tmp, tmp_h = [], []
            while nc > 0:
                km = myKMeans(self.n_clusters_list[level][pos]).fit(X[ii,:self.n_dim_list[level][pos]])
                label = km.predict(X[ii,:self.n_dim_list[level][pos]])
                tmp_h.append(Huffman().fit(label))
                tmp.append(km)
                nc = nc //2
            self.myKMeans[myhash+'_'+str(i)] = tmp
            self.Huffman[myhash+'_'+str(i)] = tmp_h
        self.Huffman[myhash+'_gidx'] = Huffman().fit(gidx)
        X, iX = X.reshape(S), iX.reshape(S)
        iX = self.RD_search_km(tX, X, gidx, level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None), True)
        X[:, :,:,:self.n_dim_list[level][pos]] -= iX[:, :,:,:self.n_dim_list[level][pos]]
        return X
    
    @Time
    def fit_one_level(self, iR, tX, level):
        myhash = 'L'+str(level)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        myLog('id=%s'%myhash)
        self.myKMeans[myhash] = myKMeans(4).fit(iR)
        gidx = self.myKMeans[myhash].predict(iR)
        for pos in range(len(self.n_dim_list[level])):
            iR = self.fit_one_level_one_pos(iR, tX, level, pos, gidx)
        return iR.reshape(self.shape[myhash])

    @Time
    def fit(self, X):
        self.cwSaab.fit(X)
        tX = self.cwSaab.transform(X)
        iR = tX[-1]
        for level in range(len(self.n_dim_list)-1, -1, -1):
            iR = self.fit_one_level(iR, tX, level)
            if level > 0:
                iR = self.cwSaab.inverse_transform_one(iR, tX[level-1], level)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, level)
        return iR

    def predict_one_level_one_pos(self, tX, X, level, pos, gidx, skip):
        myhash = 'L'+str(level)+'-P'+str(pos)
        myLog('id=%s'%(myhash))
        if myhash in skip:
            myLog('<INFO> SKIP')
            return X
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        S = X.shape
        X = X.reshape(-1, X.shape[-1])
        X = X.reshape(S)
        iX = self.RD_search_km(tX, X, gidx, level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None), False)
        X[:, :,:,:self.n_dim_list[level][pos]] -= iX
        return X
    
    #@Time
    def predict_one_level(self, tX, iR, level, skip):
        myhash = 'L'+str(level)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        for pos in range(len(self.n_dim_list[level])):
            iR = self.predict_one_level_one_pos(tX, iR, level, pos, skip)
        return iR.reshape(self.shape[myhash])

    def predict(self, X, skip=[]):
        self.buffer = {}
        self.S = []
        tX = self.cwSaab.transform(X)
        self.rX = tX.copy()
        for i in tX:
            self.S.append(i.shape)
        iR = tX[-1]
        for level in range(len(self.n_dim_list)-1, -1, -1):
            iR = self.predict_one_level(tX, iR, level, skip)
            if level > 0:
                iR = self.cwSaab.inverse_transform_one(iR, tX[level-1], level)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, level)            
        return iR

    