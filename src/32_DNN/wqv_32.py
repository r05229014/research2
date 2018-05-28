import numpy as np
from netCDF4 import Dataset
import skimage.measure


w_ = Dataset('../../../data/mjoL.w3d.nc')
qv_ = Dataset('../../../data/mjoL.qv3d.nc')

w = w_.variables['w'][0,:,:,:]
w_mean = np.mean(np.mean(w, axis=-1), axis=-1)

qv = qv_.variables['qv'][0,:,:,:]
qv_mean = np.mean(np.mean(qv, axis=-1), axis=-1)


wp = np.zeros(qv.shape)  #w'
qvp = np.zeros(qv.shape) #qv'
for i in range(70):
    wp[i] = w[i]-w_mean[i]
    qvp[i] = qv[i] - qv_mean[i]



wqv = wp*qvp 
print(wqv.shape, "wqv_shape")
# calculate wqv_mean (shape x=32 y=32)
wqv_32 = skimage.measure.block_reduce(wqv, (1,8,8), np.mean)
wqv_32 = wqv_32.reshape(1,70,32,32)
for time in range(1,1423):
    w_tmp = w_.variables['w'][time,:,:,:]
    w_tmp_mean = np.mean(np.mean(w_tmp, axis=-1),axis=-1)
    qv_tmp = qv_.variables['qv'][time,:,:,:]
    qv_tmp_mean = np.mean(np.mean(qv_tmp, axis=-1),axis=-1) 

    wp_tmp = np.zeros(qv.shape)
    qvp_tmp = np.zeros(qv.shape)
    for z in range(70):
        wp_tmp[z] = w_tmp[z] - w_tmp_mean[z]
    #    print(w_tmp_mean[i])
        qvp_tmp[z] = qv_tmp[z] - qv_tmp_mean[z]
    wqv_tmp = wp_tmp * qvp_tmp
    wqv_tmp_32 = skimage.measure.block_reduce(wqv_tmp, (1,8,8), np.mean)
    wqv_tmp_32 = wqv_tmp_32.reshape(1,70,32,32)
    wqv_32 = np.concatenate((wqv_32, wqv_tmp_32), axis=0)
    print(wqv_32.shape)
np.save('../../../data_8km_mean_y/wqv_32.npy', wqv_32)

#for i in range(1,1423):
#    w_tmp = w_.variables['w'][i:i+1,:,:,:]
#    w_tmp = skimage.measure.block_reduce(w_tmp, (1,1,8,8), np.mean)
#    w = np.concatenate((w, w_tmp), axis=0)
#    print(w.shape)

