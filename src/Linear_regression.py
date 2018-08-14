import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler
import os 
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt

def load_data():
    th = np.load('../../../data_8km_mean_X/th_8km_mean.npy')
    w = np.load('../../../data_8km_mean_X/w_8km_mean.npy')
    qv = np.load('../../../data_8km_mean_X/qv_8km_mean.npy')
    u = np.load('../../../data_8km_mean_X/u_8km_mean.npy')
    v = np.load('../../../data_8km_mean_X/v_8km_mean.npy')

    y = np.load('../../../data_8km_mean_y/wqv_32.npy')
    y = y.reshape(1423*70*32*32,1)

    th = th.reshape(1423*70*32*32,1)
    w = w.reshape(1423*70*32*32,1)
    qv = qv.reshape(1423*70*32*32,1)
    u = u.reshape(1423*70*32*32,1)
    v = v.reshape(1423*70*32*32,1)

    X = np.concatenate((th, w, qv, u, v), axis=1)
    print(X.shape)
    return X, y

X, y = load_data()
z = np.arange(70)
linear_model = LinearRegression()
linear_model.fit(X,y)

print(linear_model.coef_)
print(linear_model.intercept_ )


pre = linear_model.predict(X)
pre = pre.reshape(1423,70,32,32)
#y = y.reshape(1423,70,32,32)

y = np.load('../../../data_8km_mean_y/wqv_32.npy')
a=[0,100,200,300,400,500,600,700,800,900,1000]
b=[7,10,23,30,7,5,8,12,16,25]
c=[5,3,26,30,7,5,8,16,25,12]

img_dir = '/home/ericakcc/Desktop/research2/img/linear_model/'

for i in range(10):
    plt.figure(i)
    plt.plot(pre[a[i],:,b[i],c[i]]*2.5*10**6, z, label='Pre')
    plt.plot(y[a[i],:,b[i],c[i]]*2.5*10**6, z, label='True')
    plt.legend()
    plt.savefig(img_dir + 'img_%s' %i)
