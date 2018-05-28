import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
import os

def load_X():
    tt = np.load('../../../data_8km_mean_X/th_8km_mean.npy')
    ww = np.load('../../../data_8km_mean_X/w_8km_mean.npy')
    yy = np.load('../../../data_8km_mean_y/wqv_32.npy')
    qq = np.load('../../../data_8km_mean_X/qv_8km_mean.npy')
    uu = np.load('../../../data_8km_mean_X/u_8km_mean.npy')
    vv = np.load('../../../data_8km_mean_X/v_8km_mean.npy')
    
    print(yy.shape, '77777777777777777777')
    yy = np.swapaxes(yy,1,3)
    yy = yy.reshape(1423*32*32,70)

    #print(w.shape)
    #ww = w[392:393]
    #tt = th[392:393]
    #qq = qv[392:393]
    #uu = u[392:393]
    #vv = v[392:393]
    #yy = y[392:393]
    #print(ww.shape)
    #for i in range(393,1423):
    #    tmp = 0
    #    for j in range(32):
    #        for k in range(32):
    #            if w[i,27,j,k] > 0.5:
    #                 tmp += 1
    #    if tmp > 5 : 
    #        ww = np.concatenate((ww, w[i:i+1]), axis=0)
    #        tt = np.concatenate((tt, th[i:i+1]), axis=0) 
    #        qq = np.concatenate((qq, qv[i:i+1]), axis=0) 
    #        uu = np.concatenate((uu, u[i:i+1]), axis=0) 
    #        vv = np.concatenate((vv, v[i:i+1]), axis=0) 
    #        yy = np.concatenate((yy, y[i:i+1]), axis=0)
    #print(ww.shape)
    #print(yy.shape)

    sc = StandardScaler()
    ww = ww.reshape(1423,70*32*32)
    ww = sc.fit_transform(ww)
    
    tt = tt.reshape(1423,70*32*32)
    tt = sc.fit_transform(tt)
    
    qq = qq.reshape(1423,70*32*32)
    qq = sc.fit_transform(qq)
    
    uu = uu.reshape(1423,70*32*32)
    uu = sc.fit_transform(uu)
    
    vv = vv.reshape(1423,70*32*32)
    vv = sc.fit_transform(vv)
   
    ww = ww.reshape(1423,70,32,32,1)    
    ww = np.swapaxes(ww, 1,3)
    ww = ww.reshape(1423*32*32,70,1)

    tt = tt.reshape(1423,70,32,32,1)    
    tt = np.swapaxes(tt, 1,3)
    tt = tt.reshape(1423*32*32,70,1)
    
    qq = qq.reshape(1423,70,32,32,1)    
    qq = np.swapaxes(qq, 1,3)
    qq = qq.reshape(1423*32*32,70,1)
    
    uu = uu.reshape(1423,70,32,32,1)    
    uu = np.swapaxes(uu, 1,3)
    uu = uu.reshape(1423*32*32,70,1)
    
    vv = vv.reshape(1423,70,32,32,1)    
    vv = np.swapaxes(vv, 1,3)
    vv = vv.reshape(1423*32*32,70,1)
    
    
    X = np.concatenate((tt,ww,qq,uu,vv), axis=-1)
    X = X.reshape(1423*32*32,70*5)
    print(X.shape)
    return X, yy


X,y  = load_X()


z_ = Dataset('../../../data/mjoL.w3d.nc')
z = z_.variables['zc'][:]
if not os.path.exists('../../../img/testDNN'):
    os.mkdir('../../../img/testDNN')

model = load_model('../../../model/testDNN/weights-improvement-200-1.740e-08.hdf5')

for i in range(14220,30000):
    pre = model.predict(X[i:i+1])
    real = y[i]
    print(i)
    fig = plt.figure(i)
    plt.plot(pre[0]*2.5*10**6, z, label = 'pre')
    plt.plot(real*2.5*10**6, z, label = 'true')
    plt.title('Time = %s' %i)
    plt.xlim(-10,800)
    plt.legend()
    plt.savefig('../../../img/testDNN/testDNN_%s' %i)
    plt.close(fig)

