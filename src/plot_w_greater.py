import numpy as np

y = np.load('../../../data_8km_mean_y/wqv_32.npy')
index = np.load('./target_w_index.npy')

print(y.shape)

for i in index:
    print(i)
