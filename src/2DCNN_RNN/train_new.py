import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import *
import random
from keras import optimizers
from sklearn.preprocessing import StandardScaler
import os 

def load_X():
    tt = np.load('../../data_8km_mean_X/th_8km_mean.npy')
    ww = np.load('../../data_8km_mean_X/w_8km_mean.npy')
    yy = np.load('../../data_8km_mean_y/wqv_mean.npy')
    qq = np.load('../../data_8km_mean_X/qv_8km_mean.npy')
    uu = np.load('../../data_8km_mean_X/u_8km_mean.npy')
    vv = np.load('../../data_8km_mean_X/v_8km_mean.npy')
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
    tt = tt.reshape(1423,70,32,32,1)    
    qq = qq.reshape(1423,70,32,32,1)    
    uu = uu.reshape(1423,70,32,32,1)    
    vv = vv.reshape(1423,70,32,32,1)    
    X = np.concatenate((tt,ww,qq,uu,vv), axis=-1)
    print(X.shape)
    return X, yy

X,y  = load_X()

def build_model():
    print("Build model!!")
    model = Sequential()
    # conv1
    model.add(Conv2D(32, (3,3), use_bias=True, padding='SAME', stride=1, activation='relu', input_shape=(32,32,5)))
    model.add(Conv2D(32, (3,3), use_bias=True, padding='SAME', stride=1, activation='relu', input_shape=(32,32,5)))
    model.add(MaxPooling2D(2,2))
    # conv2
    model.add(Conv2D(64, (3,3), use_bias=True, padding='SAME', stride=1, activation='relu', input_shape=(32,32,5)))
    model.add(Conv2D(64, (3,3), use_bias=True, padding='SAME', stride=1, activation='relu', input_shape=(32,32,5)))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(256, kernel_initializer='random_uniform', use_bias=True, activation='relu'))
    model.add(Dense(128, kernel_initializer='random_uniform', use_bias=True, activation='relu'))
    model.add(Dense(64, kernel_initializer='random_uniform', use_bias=True, activation='relu'))
    model.add(Dense(1, kernel_initializer='random_uniform', use_bias=True, activation='linear'))

    model.add(LSTM(100, return_sequsnce=True, drop))
    
    





    return model

model = VGG_3D()
print(model.summary())
dirpath = "../../model/test2/"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)


filepath="../../model/test2/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = Adam, loss='mean_squared_error')
checkpoint = ModelCheckpoint(filepath, monitor='loss', 
                            save_best_only=True, mode='min',verbose=1)

model.fit(X,y, batch_size=16, epochs=1000, shuffle=True, callbacks = [checkpoint])

