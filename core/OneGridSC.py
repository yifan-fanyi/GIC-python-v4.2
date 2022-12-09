from core.util import Time, myLog
import pickle
import numpy as np
from core.util.myPCA import myPCA
from core.util.ReSample import resize
from core.util.evaluate import MSE

myLog('<FRAMEWORK> OneGrid 2022.12.09')

def pca_color(X, pca=None):
    S = X.shape
    X = X.reshape(-1, X.shape[-1])
    if pca is None:
        pca = myPCA(-1, toint=True)
        pca.fit(X)
    X = pca.transform(X)
    return X.reshape(S), pca

def pca_invcolor(X, pca=None):
    S = X.shape
    X = X.reshape(-1, X.shape[-1])
    X = pca.inverse_transform(X)
    return X.reshape(S)

def pca_color_single(X, color=None):
    pca_list = []
    tX = []
    for i in range(len(X)):
        if color is not None:
            a, _ = pca_color(X[i], color[i])
        else:
            a, b = pca_color(X[i])
            pca_list.append(b)
        tX.append(a)
    return np.array(tX), pca_list

def pca_invcolor_single(X, pca_list):
    iX = []
    for i in range(len(X)):
        iX.append(pca_invcolor(X[i], pca_list[i]))
    return np.array(iX)

def load_color(root, Y32):
    try:
        with open(root+'color_train_'+str(Y32.shape[0])+'.pkl', 'rb') as f:
            color = pickle.load(f)
    except:
        _, color = pca_color_single(Y32)
        with open(root+'color_train_'+str(Y32.shape[0])+'.pkl', 'wb') as f:
            pickle.dump(color, f, 4)
    return color
    
class OneGridSC:
    def __init__(self, grid, model_hash, model_p=None, model_q=None, model_r=None):
        self.grid = grid
        self.model_hash = model_hash
        self.model_p = model_p
        self.model_q = model_q
        self.model_r = model_r
        self.loaded = False
        self.root = './cache/'

    def load(self):
        try:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'.model', 'rb') as f:
                d = pickle.load(f)
                self.model_p, self.model_q, self.model_r = d['P'], d['Q'], d['R']
                self.loaded = True
        except:
            pass

    @Time
    def fit(self, iX, rX, color=None):
        myLog('---------------s-----------------')
        myLog('Grid=%d'%self.grid)
        self.load()
        if self.loaded == True:
            return self.predict(iX, rX, color)
        iX = resize(iX, pow(2, self.grid))
        X = rX - iX
        Y, _ = pca_color_single(X, color)
        myLog("---> P fit")
        iRp = self.model_p.fit(Y[:,:,:,:1])
        myLog("---> Q fit")
        iRq = self.model_q.fit(Y[:,:,:,1:2])
        myLog("---> R fit")
        iRr = self.model_r.fit(Y[:,:,:,2:])
        self.model_p.buffer, self.model_q.buffer, self.model_r.buffer = {}, {}, {}
        with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'.model', 'wb') as f:
            pickle.dump({'P':self.model_p, 'Q':self.model_q, 'R':self.model_r},f,4)
        self.loaded = True
        iR = np.concatenate([iRp, iRq, iRr], axis=-1)
        iX = resize(iX,pow(2, self.grid)) + pca_invcolor_single(Y - iR, color)
        myLog('---------------e-----------------')
        return iX

    @Time
    def predict(self, iX, rX, color=None, refX=None):
        myLog('---------------s-----------------')
        myLog('Grid=%d'%self.grid)
        self.load()
        iX = resize(iX, pow(2, self.grid))
        X = rX - iX
        Y, _ = pca_color_single(X, color)
        myLog("---> P predict")
        iRp = self.model_p.predict(Y[:,:,:,:1])
        myLog("---> Q predict")
        iRq = self.model_q.predict(Y[:,:,:,1:2])
        myLog("---> R predict")
        iRr = self.model_r.predict(Y[:,:,:,2:])
        iR = np.concatenate([iRp, iRq, iRr], axis=-1)
        myLog('<INFO> local MSE_p=%4.3f MSE_q=%4.3f MSE_r=%4.3f'%(MSE(iRp, np.zeros_like(iRp)),
                                                                  MSE(iRq, np.zeros_like(iRq)),
                                                                  MSE(iRr, np.zeros_like(iRr))))
        iX = resize(iX,pow(2, self.grid)) + pca_invcolor_single(Y - iR, color)
        myLog('<INFO> local MSE=%f'%MSE(rX, iX))
        if refX is not None:
            myLog('<MSE> global MSE=%f'%MSE(refX, resize(iX, refX.shape[1])))
        myLog('---------------e-----------------')
        return iX


                                                    





