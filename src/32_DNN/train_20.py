import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, MaxPooling2D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
import random
from keras import optimizers
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

def VGG_3D():
    print("Build model!!")
    model = Sequential()
#    model.add(Convolution3D(64,3, strides=(1,1,1), activation='selu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(70,32,32,5)))
#    model.add(Convolution3D(64,3, strides=(1,1,1),activation='selu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros'))
#    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))
#    model.add(Dropout(0.2))
#    model.add(BatchNormalization())

    #model.add(Convolution3D(64,3, strides=(1,1,1),activation='selu', padding='same'))
    #model.add(Convolution3D(64,3, strides=(1,1,1),activation='selu', padding='same'))
    #model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

#    model.add(Convolution3D(64,3, strides=(1,1,1),activation='relu', padding='same'))
#    model.add(Convolution3D(64,3, strides=(1,1,1),activation='relu', padding='same'))
#    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

#    model.add(Convolution3D(128,3, strides=(1,1,1),activation='selu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros'))
#    model.add(Convolution3D(128,3, strides=(1,1,1),activation='selu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros'))
#    model.add(MaxPooling3D((4,4,4), strides=(2,2,2)))
#    model.add(Dropout(0.2))
#    model.add(BatchNormalization())

#    model.add(Flatten())
    model.add(Dense(70, activation = 'selu',kernel_initializer='random_uniform',bias_initializer='zeros', input_dim=70*5))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
        
    for i in range(15):
        model.add(Dense(256, activation = 'selu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
    model.add(Dense(70, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
    return model

model = VGG_3D()
print(model.summary())
dirpath = "../../../model/testDNN/"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)


filepath="../../../model/testDNN/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
sgd = optimizers.SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss='mean_squared_error')
checkpoint = ModelCheckpoint(filepath, monitor='loss', 
                            save_best_only=False, mode='min', period=5)

model.fit(X,y, batch_size=1024, epochs=200, shuffle=True, callbacks = [checkpoint])

