import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import random
import os 

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
    th = np.load('../../../data_pooling_xy/th_pool_2.npy')
    w = np.load('../../../data_pooling_xy/w_pool_2.npy')
    print('############', w.shape)
    qv = np.load('../../../data_pooling_xy/qv_pool_2.npy')
    u = np.load('../../../data_pooling_xy/u_pool_2.npy')
    v = np.load('../../../data_pooling_xy/v_pool_2.npy')
    
    # normalize
    #sc = StandardScaler()
    #w = w.reshape(1423,70*36*36)
    #w = sc.fit_transform(w)
    
    #th = th.reshape(1423,70*36*36)
    #th = sc.fit_transform(th)
    
    #qv = qv.reshape(1423,70*36*36)
    #qv = sc.fit_transform(qv)

    #u = u.reshape(1423,70*36*36)
    #u = sc.fit_transform(u)

    #v = v.reshape(1423,70*36*36)
    #v = sc.fit_transform(v)

    # reshape to concat
    w = w.reshape(1423,70,36,36,1)   # need change
    th = th.reshape(1423,70,36,36,1)
    qv = qv.reshape(1423,70,36,36,1)
    u = u.reshape(1423,70,36,36,1)
    v = v.reshape(1423,70,36,36,1)
    X = np.concatenate((th,w,qv,u,v), axis=-1)

    print(X.shape)
    return X    

def make_right_X(X):
    x1 = X[:]
    print(x1.shape)
    tmp = np.zeros((1024*1423,70,5,5,5))  # need change(...'5','5',5)
    h=0
    for t in range(1423):
        print(t)
        for i in range(2,34): # need to change range number
            for j in range(2,33):
                tmp[h,:,:,:,:] = x1[t,:,i-2:i+3,j-2:j+3,:]  # if change size of input, need change
                #print(x1[t,:,i-2:i+3,j-2:j+3,:].shape)
                h+=1
    return tmp
X = load_X()
X = make_right_X(X)
print(X.shape)
np.save('../../../data_pooling_xy/X_all_2.npy', X)

