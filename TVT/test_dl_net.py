from keras import backend as k
from keras.models import model_from_json
import h5py
import scipy.io as sio
import numpy as np
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils import *
cr = 1/2
NMSE = []
SNR = [0, 5, 10, 15, 20]
# SNR = [0, 5, 15, 20]
for snr in SNR:
    test = 'test_snr_' + str(snr) + '_cr_' + str(cr) + '.mat'
    mat = h5py.File(test)
    x_test = mat['x_test']
    x_test = np.transpose(x_test, [2, 1, 0])
    x_test = x_test.astype('float32')  # 训练变量类型转换
    # test = 'old_375/H_test.mat'
    # mat = h5py.File(test)
    # x_test = mat['H_test']
    # x_test = np.transpose(x_test, [2, 1, 0])
    # x_test = x_test.astype('float32')  # 训练变量类型转换

    file = 'model_new_D' + '_cr_' + str(cr) + '_snr_' + str(snr)
    outfile = "result_2/saved_model/%s.json" % file
    # outfile = "old_375/%s.json" % file
    json_file = open(outfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    net = model_from_json(loaded_model_json, custom_objects={'k': k})
    outfile = 'result_2/saved_model/%s.h5' % file
    # outfile = "old_375/%s.h5" % file
    net.load_weights(outfile)

    x_hat = net.predict(x_test)

    x_re = x_test[:, 0, :]
    x_im = x_test[:, 1, :]
    x_test_c = x_re + 1j * x_im

    x_hat_re = x_hat[:, 0, :]
    x_hat_im = x_hat[:, 1, :]
    x_hat_c = x_hat_re + 1j * x_hat_im

    power = np.sum(abs(x_test_c) ** 2, axis=1)
    mse = np.sum(abs(x_test_c - x_hat_c) ** 2, axis=1)
    nmse = 10*math.log10(np.mean(mse/power))
    print('SNR is', snr, 'dB')
    print("NMSE is ", nmse, 'dB')
    NMSE.append(nmse)

sio.savemat('NMSE_DL_4_path_cr_'+ str(cr) + '.mat', mdict={'NMSE_DL_4_path_2': NMSE})