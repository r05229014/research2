import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, MaxPooling2D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import multi_gpu_model
import os 

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

    sc = StandardScaler()
    th = sc.fit_transform(th)
    w = sc.fit_transform(w)
    qv = sc.fit_transform(qv)
    u = sc.fit_transform(u)
    v = sc.fit_transform(v)

    X = np.concatenate((th, w, qv, u, v), axis=1)
    print(X.shape)
    return X, y

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
            
        return super(ModelMGPU, self).__getattribute__(attrname)


def DNN():
    print("Build model!!")
    model = Sequential()
    
    model.add(Dense(256, activation = 'relu', kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(5,)))
    for i in range(4):
        model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        #model.add(LeakyReLU(alpha=0.1))
        #model.add(BatchNormalization())
    model.add(Dense(1, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
    return model


X, y = load_data()
model = DNN()
parallel_model = ModelMGPU(model, 3)
parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
print(model.summary())
dirpath = "../../model/DNN/"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', 
                            save_best_only=False, period=5)

parallel_model.fit(X,y, batch_size=4096, epochs=150, shuffle=True, callbacks = [checkpoint])











#pre = linear_model.predict(X)
#pre = pre.reshape(1423,70,32,32)
#y = y.reshape(1423,70,32,32)

#a=[0,100,200,300,400,500,600,700,800,900,1000]
#b=[7,10,23,30,7,5,8,12,16,25]
#c=[5,3,26,30,7,5,8,16,25,12]
#
#img_dir = '/home/ericakcc/Desktop/research2/img/linear_model/'
#
#for i in range(10):
#    plt.figure(i)
#    plt.plot(pre[a[i],:,b[i],c[i]]*2.5*10**6, z, label='Pre')
#    plt.plot(y[a[i],:,b[i],c[i]]*2.5*10**6, z, label='True')
#    plt.legend()
#    plt.savefig(img_dir + 'img_%s' %i)
