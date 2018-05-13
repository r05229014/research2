import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from netCDF4 import Dataset



def load_X():
    #th = np.load('../../data_8km_mean_X/th_8km_mean.npy')
    w = np.load('../../data_8km_mean_X/w_8km_mean.npy')
    y = np.load('../../data_8km_mean_y/wqv_mean.npy')
    #qv = np.load('../../data_8km_mean_X/qv_8km_mean.npy')
    #u = np.load('../../data_8km_mean_X/u_8km_mean.npy')
    #v = np.load('../../data_8km_mean_X/v_8km_mean.npy')
    #print(w.shape)
    ww = w[392:393]
    yy = y[392:393]
    #print(ww.shape)
    for i in range(393,1423):
        tmp = 0
        for j in range(32):
            for k in range(32):
                if w[i,27,j,k] > 0.5:
                    tmp += 1
        if tmp > 5 : 
            ww = np.concatenate((ww, w[i:i+1]), axis=0)
            yy = np.concatenate((yy, y[i:i+1]), axis=0)
    print(ww.shape)
    print(yy.shape)
    ww = ww.reshape(352,70,32,32,1)    
        
    return ww, yy

X,y  = load_X()


z_ = Dataset('../../data/mjoL.w3d.nc')
z = z_.variables['zc'][:]


model = load_model('./test.h5')

for i in range(352):
    pre = model.predict(X[i:i+1])
    real = y[i]

    plt.plot(pre[0]*2.5*10**6, z, label = 'pre')
    plt.plot(real*2.5*10**6, z, label = 'true')
    #plt.xlim(0,50)
    plt.legend()
    plt.show()

