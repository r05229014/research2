import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from sklearn.preprocessing import StandardScaler

def load_data():
    th = np.load('../../../data_8km_mean_X/th_8km_mean.npy')
    w = np.load('../../../data_8km_mean_X/w_8km_mean.npy')
    qv = np.load('../../../data_8km_mean_X/qv_8km_mean.npy')
    u = np.load('../../../data_8km_mean_X/u_8km_mean.npy')
    v = np.load('../../../data_8km_mean_X/v_8km_mean.npy')

    y = np.load('../../../data_8km_mean_y/wqv_32.npy')
    #y = y.reshape(1423*70*32*32,1)

    th = th.reshape(1423*70*32*32,1)
    w = w.reshape(1423*70*32*32,1)
    qv = qv.reshape(1423*70*32*32,1)
    u = u.reshape(1423*70*32*32,1)
    v = v.reshape(1423*70*32*32,1)

    sc = StandardScaler()
    th = sc.fit_transform(th)
    w = sc.fit_transform(w)
    qv = sc.fit_transform(qv)
    u = sc.fit_transform(u)
    v = sc.fit_transform(v)

    X = np.concatenate((th, w, qv, u, v), axis=1)
    print(X.shape)
    return X, y


X, y = load_data()
model = load_model('../../model/DNN/weights-improvement-030-3.396e-09.hdf5')

z = np.arange(70)
print(z)
a=[0,100,200,300,400,500,600,700,800,900,1000]
b=[7,10,23,30,7,5,8,12,16,25]
c=[5,3,26,30,7,5,8,16,25,12]

img_dir = '/home/ericakcc/Desktop/research2/img/DNN/'
pre = model.predict(X, batch_size=2048)
pre = pre.reshape(1423,70,32,32,1)
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

for i in range(10):
    
    plt.figure(i)
    plt.plot(pre[a[i],:,b[i],c[i]]*2.5*10**6, z, label='Pre')
    plt.plot(y[a[i],:,b[i],c[i]]*2.5*10**6, z, label='True')
    plt.legend()
    plt.savefig(img_dir + 'img_%s' %i)
