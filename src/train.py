import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.callbacks import *
import random


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

def VGG_3D():
    print("Build model!!")
    model = Sequential()
    model.add(Convolution3D(64,3, strides=(1,1,1), activation='relu', padding='same', input_shape=(70,32,32,1)))
    model.add(Convolution3D(64,3, strides=(1,1,1),activation='relu', padding='same'))
    model.add(MaxPooling3D((8,8,8), strides=(2,2,2)))

    #model.add(Convolution3D(64,3, strides=(1,1,1),activation='selu', padding='same'))
    #model.add(Convolution3D(64,3, strides=(1,1,1),activation='selu', padding='same'))
    #model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

#    model.add(Convolution3D(64,3, strides=(1,1,1),activation='relu', padding='same'))
#    model.add(Convolution3D(64,3, strides=(1,1,1),activation='relu', padding='same'))
#    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

    model.add(Convolution3D(128,3, strides=(1,1,1),activation='relu', padding='same'))
    model.add(Convolution3D(128,3, strides=(1,1,1),activation='relu', padding='same'))
    model.add(MaxPooling3D((8,8,8), strides=(2,2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.7))
    model.add(Dense(70, activation = 'linear'))

    return model

model = VGG_3D()
print(model.summary())

model.compile(optimizer = 'adam', loss='mean_squared_error')
checkpoint = ModelCheckpoint(filepath='./test.h5', monitor='loss', 
                                 save_best_only=True)

model.fit(X,y, batch_size=16, epochs=100, shuffle=True, callbacks = [checkpoint])


