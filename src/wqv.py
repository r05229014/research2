import numpy as np
from netCDF4 import Dataset


w_ = Dataset('../../data/mjoL.w3d.nc')
qv_ = Dataset('../../data/mjoL.qv3d.nc')

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
wqv_mean = np.mean(np.mean(wqv,axis=-1),axis=-1)
wqv_mean = wqv_mean.reshape(1,70)
for time in range(1,1423):
    w_tmp = w_.variables['w'][i,:,:,:]
    w_tmp_mean = np.mean(np.mean(w_tmp, axis=-1),axis=-1)

    qv_tmp = qv_.variables['qv'][i,:,:,:]
    qv_tmp_mean = np.mean(np.mean(qv_tmp, axis=-1),axis=-1) 

    wp_tmp = np.zeros(qv.shape)
    qvp_tmp = np.zeros(qv.shape)
    for z in range(70):
        wp_tmp[i] = w_tmp[i] - w_tmp_mean[i]
        qvp_tmp[i] = qv_tmp[i] - qv_tmp_mean[i]
    wqv_tmp = wp_tmp * qvp_tmp
    wqv_tmp_mean = np.mean(np.mean(wqv_tmp,axis=-1),axis=-1)
    wqv_tmp_mean = wqv_tmp_mean.reshape(1,70)
    wqv_mean = np.concatenate((wqv_mean, wqv_tmp_mean), axis=0)
    print(wqv_mean.shape)

np.save('../../data_8km_mean_y/wqv_mean.npy', wqv_mean)

#for i in range(1,1423):
#    w_tmp = w_.variables['w'][i:i+1,:,:,:]
#    w_tmp = skimage.measure.block_reduce(w_tmp, (1,1,8,8), np.mean)
#    w = np.concatenate((w, w_tmp), axis=0)
#    print(w.shape)

