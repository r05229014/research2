import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import random
import os 
import psutil
process = psutil.Process(os.getpid())

def load_y():
    y = np.load('../../../data_8km_mean_y/wqv_32.npy')
    
    new_y = np.zeros((32*32*1423,70))
    j = 0
    for t in range(1423):
        print(j)
        for x in range(32):
            for y_ in range(32):
                new_y[j] = y[t,:,x,y_]
                j +=1
    print(y.shape)
    return new_y

def load_X():
    th = np.load('../../../data_pooling_xy/th_pool.npy')
    w = np.load('../../../data_pooling_xy/w_pool.npy')
    qv = np.load('../../../data_pooling_xy/qv_pool.npy')
    u = np.load('../../../data_pooling_xy/u_pool.npy')
    v = np.load('../../../data_pooling_xy/v_pool.npy')
    
    # normalize
    sc = StandardScaler()
    w = w.reshape(1423,70*34*34)
    w = sc.fit_transform(w)
    
    th = th.reshape(1423,70*34*34)
    th = sc.fit_transform(th)
    
    qv = qv.reshape(1423,70*34*34)
    qv = sc.fit_transform(qv)

    u = u.reshape(1423,70*34*34)
    u = sc.fit_transform(u)

    v = v.reshape(1423,70*34*34)
    v = sc.fit_transform(v)

    # reshape to concat
    w = w.reshape(1423,70,34,34,1)
    th = th.reshape(1423,70,34,34,1)
    qv = qv.reshape(1423,70,34,34,1)
    u = u.reshape(1423,70,34,34,1)
    v = v.reshape(1423,70,34,34,1)
    X = np.concatenate((th,w,qv,u,v), axis=-1)

    print(X.shape)
    return X    

def make_right_X(X):
    x1 = X[1200:1423]
    print(x1.shape)
    tmp = np.zeros((1024*(1423-1200),70,3,3,5))
    h=0
    for t in range(1423-1200):
        for i in range(1,33):
            for j in range(1,33):
                tmp[h,:,:,:,:] = x1[t,:,i-1:i+2,j-1:j+2,:] 
                h+=1
    return tmp
X = load_X()
print(process.memory_info().rss*10**-6)
X = make_right_X(X)
print(process.memory_info().rss*10**-6)
print(X.shape)
np.save('../../../data_pooling_xy/X_1200_1423.npy', X)

