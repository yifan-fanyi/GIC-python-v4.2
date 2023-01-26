import numpy as np
from core.util.myPCA import myPCA
from core.util.myKMeans import myKMeans
from util import Shrink, invShrink

class PQR:
    def __init__(self, win):
        self.win = win
        self.myPCA = []
        self.colorM = myKMeans(256)

    def fit(self, X):
        X = Shrink(X, self.win)
        mX = np.mean(X, axis=-1, keepdims=True)  
        self.colorM.fit(mX)
        label = self.colorM.predict(mX)
        imX = self.colorM.inverse_predict(label)   
        print(np.mean(np.square(mX-imX)))   
        X -= imX
        X = invShrink(X, self.win)
        X = X.reshape(-1, X.shape[-1])
        self.myPCA.fit(X)
        return self

    def transform(self, X):
        X = Shrink(X, self.win)
        mX = np.mean(X, axis=-1, keepdims=True)        
        X -= mX
        X = invShrink(X, self.win)
        tX = self.myPCA.transform(X)
        return mX, tX
    
    def inverse_transform(self, mX, tX):
        X = self.myPCA.inverse_transform(tX)
        X = Shrink(X, self.win)
        X += mX
        X = invShrink(X, self.win)
        return X



if __name__ == "__main__":
    import cv2
    x = cv2.imread('/Users/alex/Desktop/proj/data/train1024/9.png')
    x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2]).astype('float32')
    pqr = PQR(16)
    pqr.fit(x)
    mx, tx = pqr.transform(x)
    ix = pqr.inverse_transform(mx, tx)
    print(mx)
    print(np.mean(np.abs(x-ix)))

