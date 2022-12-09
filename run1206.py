from core.VQ import *
from core.util.evaluate import *
from core.data import *
import numpy as np
from core.util import *
import warnings
warnings.filterwarnings("ignore")
from core.OneGridSC import OneGridSC, load_color
from time import gmtime, strftime, localtime
from core.spatialKM import spatialKM
root = './'
fit32 = True
fit256 = True
save_src = False
testOnly = False
myhash = None

myLog('------------------------------------------------------')
myLog('---START RUN---')
if testOnly == True:
    os.system('mv -r ./core ./core_tmp')
    os.system('cp -r ./cache/core_'+myhash+' ./core')
    myLog('load ID_hash=%s'%myhash)

if (fit32 == True or fit256 == True) and save_src == True:
    myhash = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    myLog('ID_hash=%s'%myhash)
    os.system('cp -r ./core ./cache/core_'+myhash)
g8_skip_p = ['L0-P0']
g3_hash, g5_hash, g8_hash = 'unit_'+myhash, 'unit_'+myhash, 'unit_'+myhash
myLog('g3_hash=%s g5_hash=%s g8_hash=%s'%(g3_hash,g5_hash,g8_hash))

g3VQp = VQ(n_clusters_list=[[1024],
                            [4,1024],
                            [2,4,4,1024]], 
            win_list=[2,2,2], 
            n_dim_list=[[4],
                        [3,4],
                        [1,2,3,4]])
g3VQq = VQ(n_clusters_list=[[256],
                            [4,16,128]], 
            win_list=[4,2], 
            n_dim_list=[[16],
                        [1, 3,4]], 
            enable_skip={'L0-P0':[False,10]})
g3VQr = VQ(n_clusters_list=[[16],
                            [4,64]], 
            win_list=[4,2], 
            n_dim_list=[[16],
                        [2,4]], 
            enable_skip={'L0-P0':[False,10]})

g5VQp = VQ(n_clusters_list=[[32],
                            [1024],
                            [16,1024],
                            [16,1024],
                            [16,64,1024]], 
            win_list=[2,2,2,2,2], 
            n_dim_list=[[4],
                        [4],
                        [3,4],
                        [3,4],
                        [2,3,4]], 
            enable_skip={'L0-P0':[True,1600],'L1-P0':[True,500],'L1-P0':[True,100]})
g5VQq = VQ(n_clusters_list=[[128],
                            [128],
                            [128],
                            [128],      
                            [128]], 
            win_list=[2,2,2,2,2], 
            n_dim_list=[[4],
                        [4],
                        [4],
                        [4],
                        [4]], 
            enable_skip={'L0-P0':[True,1600],'L1-P0':[True,1000],'L1-P0':[True,600]})
g5VQr = VQ(n_clusters_list=[[4],
                            [4],
                            [4,8]], 
            win_list=[8,2,2], 
            n_dim_list=[[64],
                        [4],
                        [2,4]], 
            enable_skip={'L0-P0':[False,100]})

g8VQp = VQ(n_clusters_list=[[4,1024], 
                            [8,1024], 
                            [8,1024], 
                            [8,1024], 
                            [8, 1024], 
                            [8, 1024], 
                            [8, 1024], 
                            [8, 1024]], 
            win_list=[2, 2, 2, 2, 2, 2, 2, 2], 
            n_dim_list=[[4, 48],
                        [4, 12], 
                        [4, 12], 
                        [4, 12], 
                        [4, 12], 
                        [4, 12], 
                        [4, 12], 
                        [4, 12]], 
            enable_skip={'L0-P0': [True, 6500],'L0-P1': [True, 6500],
                        'L1-P0': [True, 6500], 'L1-P1': [True, 6500], 
                        'L2-P0': [True, 4500],'L2-P1': [True, 4500],
                        'L3-P0': [True, 1500], 'L3-P1': [True, 1500],
                        'L4-P0': [True, 500],'L4-P1': [True, 500],
                        'L5-P0': [True, 100],'L5-P1': [True, 100]},
            transform_split=2)
g8VQq = VQ(n_clusters_list=[[512], 
                            [512], 
                            [512], 
                            [512], 
                            [512]], 
            win_list=[4,4, 4, 2, 2], 
            n_dim_list=[[64], [16], [16], [4], [4]], 
            enable_skip={'L0-P0': [True, 1500],
                        'L1-P0': [True, 1000],
                        'L2-P0': [True, 700],
                        'L3-P0': [True, 700],
                        'L3-P1': [True, 600],
                        'L4-P0': [True, 300]})
g8VQr = VQ(n_clusters_list=[[4], [4], [4], [8]], 
            win_list=[16, 4, 2, 2], 
            n_dim_list=[[256], [16], [4], [4]], 
            enable_skip={'L0-P0': [False, 300]})

g3 = OneGridSC(grid=3, model_hash=g3_hash, model_p=g3VQp, model_q=g3VQq, model_r=g3VQr)
g5 = OneGridSC(grid=5, model_hash=g5_hash, model_p=g5VQp, model_q=g5VQq, model_r=g5VQr)
g8 = OneGridSC(grid=8, model_hash=g8_hash, model_p=g8VQp, model_q=g8VQq, model_r=g8VQr)
g8skm = spatialKM(grid=8, model_hash=g8_hash, win_list=[8,2], n_clusters_list=[2048, 2048], skip_TH_list=[600, 1200])

tmp = load(Rtype='test', ct=-1, size=[1024,256,32,8])
Yt, Y256t, Y32t, Y8t = tmp[0], tmp[1], tmp[2], tmp[3]
colort = load_color(root, Y32t)
if fit32 == True:
    Y8, Y32 = load(Rtype='small', ct=[190,1000], size=None)
    color = load_color(root, Y32)

    iY8 = g3.fit(np.zeros_like(Y8), Y8, color)
    iY8t = g3.predict(np.zeros_like(Y8t), Y8t, colort, Yt)

    iY32 = g5.fit(iY8, Y32, color)
    iY32t = g5.predict(iY8t, Y32t, colort, Yt)
    Y8, Y32, iY8, iY32, color = None, None, None, None, None

if fit256 == True:
    tmp = load(Rtype='train', ct=20000, size=[1024,256,32,8])
    Y, Y32, Y8 = tmp[0], tmp[1], tmp[2]
    color = load_color(root, Y)
    tmp = None

    iY8 = g3.predict(np.zeros_like(Y8), Y8, color)
    iY8t = g3.predict(np.zeros_like(Y8t), Y8t, colort, Yt)

    iY32 = g5.predict(iY8, Y32, color)
    iY32t = g5.predict(iY8t, Y32t, colort, Yt)

    iY256 = g8.fit(iY32, Y, color)
    iY256t = g8.predict(iY32t, Y256t, colort, Yt)

    iY256 = g8skm.fit(iY256, Y)
    iY256t = g8skm.predict(iY256t, Y256t, Yt)
if testOnly == True:
    iY8t = g3.predict(np.zeros_like(Y8t), Y8t, colort, Yt)
    iY32t = g5.predict(iY32t, Y32t, colort, Yt)
    iY256t = g8.predict(iY32t, Y256t, colort, Yt)
    iY256t = g8skm.predict(iY256t, Y256t, Yt)
myLog('---END RUN---')
myLog('------------------------------------------------------')
if testOnly == True:
    os.system('rm -rf ./core ')
    os.system('mv ./core_tmp ./core')

if (fit32 == True or fit256 == True) and save_src == True:
    os.system('mv ./log/log.txt ./log/log_'+myhash+'.txt')