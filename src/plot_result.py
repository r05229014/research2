import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import os

plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_y():
    print("loading...y")
    y = np.load('../../../data_8km_mean_y/wqv_32.npy')
    y = y[:]    
    new_y = np.zeros((32*32*1423,70))
    j = 0
    for t in range(1423):
        print(t)
        for x in range(32):
            for y_ in range(32):
                new_y[j] = y[t,:,x,y_]
                j +=1
    return new_y

def load_X():
    print("loading...X")
    x1 = np.load('../../../data_pooling_xy/X_all.npy')
    x1 = x1[:,:,:,:,0:1]
    print(x1.shape)
        
    return x1

img_dir = '../../img/conv3d_test3_nor_th/' 
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

y = load_y()
X = load_X()
model = load_model('../../model/nor_onlyuse_th/weights-improvement-005-1.901e-08.hdf5')
pre = model.predict(X[100000:110000], batch_size=1000)
print("pre shape : ", pre.shape)
print(y.shape[0])
for i in range(10000):  #pre = model.predict(X[i:i+1], batch_size=1)
    #plt.figure(i)
    print(i)
    plt.plot(y[i]*2.5*10**6 , label = 'True')
    plt.plot(pre[i]*2.5*10**6 , label='pre')
    plt.legend()
    plt.savefig(img_dir + '%s' %i)
    plt.close()
