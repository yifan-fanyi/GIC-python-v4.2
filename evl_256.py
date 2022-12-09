import cv2
import os
import numpy as np
import pickle
from core.util.evaluate import MSE, PSNR
from core.util.ReSample import ReSample
import matplotlib.pyplot as plt

def myFilter(res):
    tmp = []
    last = res[0, 0]
    for i in range(1, len(res)):
        if res[i, 0] < last:
            tmp.append(res[i])
            last = res[i, 0]
    return np.array(tmp).reshape(-1,3)

def resize(Y, S):
    while Y.shape[1] < S:
        Y = ReSample.resample(Y, 1/2)
    return Y

def load_imgs_bpg(folder, suflex, count=186):
    img, bpp = [], []
    for i in range(count):
        if suflex is not None:
            x = cv2.imread(folder+'/'+str(i)+suflex+'.png')
        else:
            x = cv2.imread(folder+'/'+str(i)+'.png')
        img.append(x)
        if suflex is not None:
            b = os.stat(folder+'/'+str(i)+suflex+'.bpg').st_size*8
            bpp.append(b)
    return np.array(img), np.array(bpp)

def load_imgs_jpg(folder, suflex, count=186):
    img, bpp = [], []
    for i in range(count):
        if suflex is not None:
            x = cv2.imread(folder+'/'+str(i)+suflex+'.bmp')
        else:
            x = cv2.imread(folder+'/'+str(i)+'.bmp')
        img.append(x)
        if suflex is not None:
            b = os.stat(folder+'/'+str(i)+suflex+'.jpg').st_size*8
            bpp.append(b)
    return np.array(img), np.array(bpp)

def evl_one_bpg(gt, folder, Q):
    res = []
    for q in range(Q[0], Q[1]):
        img, bpp = load_imgs_bpg(folder, '_'+str(q))
        img = resize(img, gt.shape[1])
        res.append([MSE(gt, img), PSNR(gt, img), np.mean(bpp)/(float)(gt.shape[1])**2])
    return res
def evl_one_jpg(gt, folder, Q):
    res = []
    for q in range(Q[0], Q[1]):
        img, bpp = load_imgs_jpg(folder, '_'+str(q))
        img = resize(img, gt.shape[1])
        res.append([MSE(gt, img), PSNR(gt, img), np.mean(bpp)/(float)(gt.shape[1])**2])
    return res

def evl_bpg():
    folders = ['/Users/alex/Desktop/proj/bpg_res/bpgtest32/420',
                '/Users/alex/Desktop/proj/bpg_res/bpgtest64/420',
                '/Users/alex/Desktop/proj/bpg_res/bpgtest128/420',
               '/Users/alex/Desktop/proj/bpg_res/bpgtest256/420']
    gt, _ = load_imgs_bpg('/Users/alex/Desktop/proj/data/test256', None)
    res = []
    for i in range(len(folders)):
        if i == len(folders)-1:
            res += evl_one_bpg(gt, folders[i], [40, 52])
        else:
            res += evl_one_bpg(gt, folders[i], [1, 52])
    res = np.array(res).reshape(-1, 3)
    idx = np.argsort(res[:, 2])
    res = res[idx]
    return myFilter(res)

def evl_jpg():
    folders = ['/Users/alex/Desktop/proj/jpg_res/jpgtest256/420']
    '''
    ['/Users/alex/Desktop/proj/jpg_res/jpgtest32/420',
                '/Users/alex/Desktop/proj/jpg_res/jpgtest64/420',
                '/Users/alex/Desktop/proj/jpg_res/jpgtest128/420',
               '/Users/alex/Desktop/proj/jpg_res/jpgtest256/420']
    '''
    gt, _ = load_imgs_jpg('/Users/alex/Desktop/proj/data/test256bmp', None)
    res = []
    for i in range(len(folders)):
        if i == len(folders)-1:
            res += evl_one_jpg(gt, folders[i], [1, 41])
        else:
            res += evl_one_jpg(gt, folders[i], [1, 100])
    res = np.array(res).reshape(-1, 3)
    idx = np.argsort(res[:, 2])
    res = res[idx]
    return myFilter(res)

def _plot(x, label, axis=0):
    x = sumall(x)
    if axis ==1:
        plt.plot(x[:,2], mse2psnr(x[:,0]), label=label)
    else:
        plt.plot(x[:,2], x[:,axis], label=label)

def sumall(a):
    a = np.array(a).reshape(-1, 3)
    for i in range(1,a.shape[0]):
        a[i, -1] += a[i-1, -1] 
    return a

def load():
    try:
        with open('/Users/alex/Desktop/proj/evl/evl_bpg_256.pkl', 'rb') as f:
            bpg = pickle.load(f)
    except:
        bpg = evl_bpg()
        with open('/Users/alex/Desktop/proj/evl/evl_bpg_256.pkl', 'wb') as f:
            pickle.dump(bpg, f)

    try:
        with open('/Users/alex/Desktop/proj/evl/evl_jpg_256.pkl', 'rb') as f:
            jpg = pickle.load(f)
    except:
        jpg = evl_jpg()
        with open('/Users/alex/Desktop/proj/evl/evl_jpg_256.pkl', 'wb') as f:
            pickle.dump(jpg, f)
    return myFilter(jpg), myFilter(bpg)

# evl result
log_2022_07_01_10_09_24 = [1648.348, 16.8008, 0.0030308, 1457.896, 17.403, 0.000473, 1219.731, 18.237, 0.007245, 1124.312, 18.6331, 0.000422, 985.312, 19.2753, 0.001872, 902.624, 19.6522, 0.0010844761904761904, 767.568, 20.3381, 0.004059904761904762, 641.507, 21.0756, 0.011389619047619048, 574.76, 21.5528, 0.003715, 494.067, 22.1736, 0.012501, 
#475.771, 22.3247, 0.003026, 419.223, 22.8225, 0.014376, 399.669, 23.0704, 0.0054999882352941174, 363.869, 23.4552, 0.012675988235294117, 302.426, 24.1503, 0.03459498823529412, 240.784, 24.9113, 0.053131988235294116
]

log_vq0810_full_run = [2563.558,14.6137,0.000206,
                        1716.896,16.5574,0.000865,
                        1283.805,18.0367,0.003515,
                        1222.814,18.2915,0.000171,
                        1110.551,18.7611,0.000740,
                        918.257,19.6653,0.003327,
                        776.952,20.2561,0.003030+0.000842,
                        769.372,20.3062,0.000401+0.000031,
                        719.074,20.6254,0.002255+0.000154,
                        605.929,21.3830,0.007611+0.000929,
                        478.165,22.3065,0.017242+0.003952,]
                        #297.139,24.3975,0.080603+0.016074]

log_0805 = [2648.645, 14.4423, 0.000178,
            1749.398, 16.4610, 0.000798,
            1292.312, 17.9948, 0.003227,
            1235.366, 18.2343, 0.000147,
            1126.701, 18.6792, 0.000676,
            935.633, 19.5601, 0.003059,
            794.440, 20.1395, 0.002877+0.000769,
            789.301, 20.1696, 0.000533,
            751.165, 20.4027, 0.002278, #G8 64
             751.165, 20.4027,0.002278
            ]
                        
def Slop(bpp, mse):
    slop = []
    for i in range(1, len(bpp)):
        s = (mse[i] - mse[i-1]) / (bpp[i] - bpp[i-1])
        slop.append([(bpp[i] + bpp[i-1])/2, s, (mse[i] + mse[i-1])/2])
    return np.array(slop).reshape(-1,3)

def mse2psnr(val):
    return 10*np.log10(255**2/val)




if __name__ == "__main__":
    jpg, bpg = load()

    plot_psnr = 0
    plot_mse = 1
    plot_slop = 0
    if True == plot_psnr:
        plt.figure(figsize=(6,8))
        _plot(log_2022_07_01_10_09_24, label='Ours SPIE', axis=1)
        #_plot(log_vq0810_full_run, label='log_vq0810_full_run', axis=1)
        plt.plot(bpg[:25, 2], mse2psnr(bpg[:25, 0]), label='BPG-420')
        #plt.plot(jpg[:15, 2], jpg[:15, 1], label='JPG-420')
        plt.grid()
        a = np.array([[0.0041764782321068545,1274.247576805853],
                        [0.000206+0.000865+0.003515+0.000171+0.000740+0.003327,918.257],
                        [0.0041764782321068545+0.00924370878486223,780.8223204527278],
                        [0.000206+0.000865+0.003515+0.000171+0.000740+0.003327+0.000842+0.000031+0.002255+0.000154+0.007611+0.000929,605.929],
                       [0.000206+0.000865+0.003515+0.000171+0.000740+0.003327+0.000842+0.000031+0.002255+0.000154+0.007611+0.000929+0.017242+0.003952,478.165],
                        #[0.0041764782321068545+0.00924370878486223+0.010785256662676412,691.4297630278037],
                        #[0.0041764782321068545+0.00924370878486223+0.021637865292128695,604.7686892850725],
                        #[0.0041764782321068545+0.00924370878486223+0.05316859419627856, 425.35774684392925],
                        #[0.0041764782321068545+0.00924370878486223+0.07637696625084005, 345.2710423527456],
                        
                        #[0.0041764782321068545+0.00924370878486223+0.07928811350176411, 320.9058942350584],
                        #[0.0041764782321068545+0.00924370878486223+0.08484042588100639,317.67033463026377]
                        ])


        plt.plot(a[:,0], mse2psnr(a[:,1]), label='Ours New')
        plt.xlabel('BPP',fontsize=16)
        plt.ylabel('PSNR',fontsize=16)
        plt.legend()#fontsize=16)
        plt.savefig('performance.pdf',bbox_inches='tight')
        plt.show()
    if True == plot_mse:
        plt.figure(figsize=(6,9))
        #_plot(log_2022_07_01_10_09_24, label='Ours SPIE', axis=0) # 2022_07_01_10_09_24
        #_plot(log_vq0810_full_run, label='Ours Improved', axis=0) # log_vq0810_full_run
        plt.plot(bpg[:, 2], bpg[:, 0])
        plt.plot(bpg[:25, 2], bpg[:25, 0], label='BPG-420')
        print(bpg[:,2])
        print(bpg[:,0])
        plt.plot(jpg[:15, 2], jpg[:15, 0], label='JPG-420')
       
        a = np.array([[0.0041764782321068545,1274.247576805853],
                        [0.0041764782321068545+0.00924370878486223,780.8223204527278],
                        [0.0041764782321068545+0.00924370878486223+0.09179211688298052,318.4359490646365]])
        #plt.scatter(a[:,0], a[:,1], label='0924b.model')
        a = np.array([[80/256/256+0.0041764782321068545,1274.247576805853],
                        [80/256/256+0.000206+0.000865+0.003515+0.000171+0.000740+0.003327,918.257],
                        [80/256/256+0.0041764782321068545+0.00924370878486223,780.8223204527278],
                        [80/256/256+0.000206+0.000865+0.003515+0.000171+0.000740+0.003327+0.000842+0.000031+0.002255+0.000154+0.007611+0.000929,605.929],
                       [80/256/256+0.000206+0.000865+0.003515+0.000171+0.000740+0.003327+0.000842+0.000031+0.002255+0.000154+0.007611+0.000929+0.017242+0.003952,478.165],                        ])
        #plt.plot(a[:,0], a[:,1], label='Ours New')
        _plot(log_2022_07_01_10_09_24, label='Ours New', axis=0)

        #plt.scatter(0.004182548933131721, 1272, label='G3_0930a')
        #plt.scatter(0.004182548933131721+0.012576130922379033, 719.44, label='G3_0930a+G5_1108a')
        #plt.scatter(0.004182548933131721+0.012576130922379033+0.004679361979166667, 642, label='G3_0930a+G5_1108a+G8_1108a')
        #plt.scatter(0.004182548933131721+0.012576130922379033+0.004679361979166667+0.065, 463, label='secondary VQ')
        #plt.scatter(0.004182548933131721+0.012576130922379033+0.004679361979166667+0.0332, 475, label='G3_0930a+G5_1108a+G81108i')
        #lt.scatter(0.004182548933131721+0.012576130922379033+0.09, 370, label='G3_0930a+G5_1108a+G8_1108x')
        #plt.scatter(0.004182548933131721+0.012576130922379033+0.06436444354313677, 389.94, label='G3_0930a+G5_1108a+G8_1108y')
        #plt.scatter(0.004182548933131721+0.012576130922379033+0.016, 586, label='G3_0930a+G5_1108a+G8_1108y to 16x16')
        #plt.scatter(0.004182548933131721+0.012+0.15, 258, label='G3_0930a+G5_1108a+G8_1108y to 16x16 + spatial 8x8 &4x4')
        #plt.scatter(0.004182548933131721+0.012+0.016670473160282258+0.07737174085391466,310.1521242647996, label='G3_0930a+G5_1108a+G8_1108y to 16x16 + spatial 4x4')

        a = [0.004182548933131721,1272.8182781588646,
0.004182548933131721+0.013523988128990256, 694.7074278417881,
0.004182548933131721+0.013523988128990256+0.021480765393985215,530.611677174816,
0.004182548933131721+0.013523988128990256+0.021480765393985215+0.0026362429382980512+0.00971262942078293,443.7055282546555,
0.004182548933131721+0.013523988128990256+0.021480765393985215+0.0026362429382980512+0.00971262942078293+0.004664841518607190+0.014393796202956988,396.1272189258911,
0.004182548933131721+0.013523988128990256+0.021480765393985215+0.004852212885374664 +0.02087878155451949,397.3625146947287,
0.004182548933131721+0.013523988128990256+0.021480765393985215+0.004852212885374664 +0.02087878155451949+0.008085025254116264+ 0.02635857366746472,333.43108894987387

]
        a = np.array(a).reshape(-1,2)
        plt.scatter(a[:, 0], a[:, 1], label='New')
        #plt.scatter(0.0041764782321068545+0.00924370878486223+0.05741439327116935, 427.66401094994546,label='0924b1.model')
        plt.scatter(0.004182548933131721+0.013523988128990256+0.021480765393985215+0.004852212885374664 +0.02087878155451949,397.3625146947287, color='k')
        plt.scatter(0.004182548933131721+0.013523988128990256+0.021480765393985215+0.004852212885374664 +0.02087878155451949+0.008085025254116264+ 0.02635857366746472,333.43108894987387,color='k')
        plt.grid()
        plt.xlabel('BPP')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig('mse.pdf',bbox_inches='tight')
        plt.show()
    






















    
    if True == plot_slop:
        from sklearn.ensemble import RandomForestRegressor
        b = [[0.006314185357862903, -152140.41000682986, 1218.530709201717], 
            [0.006824124243951613, -124246.3988240069, 1148.472417359711], 
            [0.007411505586357527, -106850.6735106749, 1081.0180717249498], 
            [0.008141630439348119, -80354.30290373896, 1013.9329497634724], 
            [0.008907851352486559, -62010.965715831786, 958.852600001947], 
            [0.009611396379368281, -52390.53037054402, 918.5903567379094], 
            [0.010683449365759408, -46197.38943396228, 863.5652400738022], 
            [0.01176370600218414, -31656.573440942186, 823.0457649504413], 
            [0.015179049584173387, -25607.848632261783, 707.7728120673942], 
            [0.017312654884912633, -20706.116906305753, 658.8410620467209], 
            [0.0195729245421707, -16166.785680435038, 617.0162855715735],
            [0.02174065702704973, -12212.07979836644, 586.208077885344], 
            [0.027886175340221774, -7679.583660422265, 516.5902236063421], 
            [0.03730691889280914, -6453.42669077723, 434.8923486552359], 
            [0.04322634461105511, -5267.417558340977, 400.5108328378328], 
            [0.068287839171707, -4459.612378345479, 312.0001969559645], 
            [0.09105526503696236, -3916.376665159726, 262.7917005614141], 
            [0.105549802062332, -3403.254309401331, 239.87749578277695], 
            [0.11778406943044355, -1326.1152606054145, 217.42562339810058], 
            [0.13625770486811156, -1006.9657516606874, 195.87564118361388], 
            [0.15754453597530244, -861.4923370734489, 176.19336854941528], 
            [0.18168492471018144, -645.2721275363891, 158.01096951064244], 
            [0.2097122028309812, -557.0959670057382, 141.33093068932976], 
            [0.2411170877436156, -397.88215952482017, 126.2982429039521], 
            [0.2772315240675403, -366.2460741305866, 112.58210002393278]]
        b = np.array(b)
        svr = RandomForestRegressor().fit(b[:,0].reshape(-1, 1), b[:,1].reshape(-1,1))
        plt.plot(b[:,0], svr.predict(b[:,0].reshape(-1,1)).reshape(-1),label='fit')
        plt.plot(b[:, 0], b[:, 1], label='BPG-420')
        plt.legend()
        plt.show()
        import pickle
        svrr = RandomForestRegressor().fit(b[:,2].reshape(-1, 1), b[:,1].reshape(-1,1))

        #with open('/Users/alex/Desktop/rd_model.pkl', 'wb') as f:
        #    pickle.dump({'slop_bpp':svr, 'slop_mse':svrr}, f)
        plt.plot(b[:,2], svrr.predict(b[:,2].reshape(-1,1)).reshape(-1),label='fit')
        plt.plot(b[:, 2], b[:, 1], label='BPG-420')
        plt.legend()
        plt.show()
    
