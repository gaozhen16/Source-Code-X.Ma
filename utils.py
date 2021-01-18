from keras import backend as k
from keras.layers import Layer, Conv2D, Activation, BatchNormalization, Subtract, add, LeakyReLU, Reshape
import numpy as np
import math
import cmath
import scipy
import keras
import tensorflow as tf


num = 2
m = 16
n = 16
N_BS = m*n
N_ms = 1
cr = 1/2
mtx_v = N_BS
mtx_h = int(N_BS*cr)
output_dim = int(N_BS*cr)
q = 2
d_ant = 1/2
j = cmath.sqrt(-1)
pi = math.pi


class MatMul(Layer):
    def __init__(self, output_dim, mtx_v, mtx_h, **kwargs):
        self.output_dim = output_dim
        self.mtx_v = mtx_v
        self.mtx_h = mtx_h
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.mtx_v = k.variable()
        self.matrix = self.add_weight(name='matrix',
                                      shape=(self.mtx_v, self.mtx_h),
                                      initializer='RandomNormal',
                                      trainable=True)
        super(MatMul, self).build(input_shape)

    def call(self, x):
        return k.dot(x, self.matrix)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'mtx_v': self.mtx_v,
            'mtx_h': self.mtx_h,
        }
        base_config = super(MatMul, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class dictionary(Layer):
#     def __init__(self, S_init, F_init, D_init, **kwargs):
#         self.S_init = S_init
#         self.F_init = F_init
#         self.D_init = D_init
#         super(dictionary, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.S = self.add_weight(name='S',
#                                  shape=self.S_init.shape,
#                                  initializer=keras.initializers.Constant(value=self.S_init),
#                                  trainable=False)
#         self.F = self.add_weight(name='F',
#                                  shape=self.F_init.shape,
#                                  initializer='Identity',
#                                  # initializer=keras.initializers.Constant(value=self.F_init),
#                                  trainable=False)
#         self.a = self.add_weight(name='a',
#                                  shape=(2, 4),
#                                  dtype=tf.complex128,
#                                  initializer='Zeros',
#                                  regularizer=None,
#                                  trainable=False)
#         self.D = self.add_weight(name='D',
#                                  shape=self.D_init.shape,
#                                  initializer=keras.initializers.Constant(value=self.D_init),
#                                  trainable=False)
#         self.S = tf.cast(self.S, tf.complex128)
#         self.F = tf.cast(self.F, tf.complex128)
#         self.D = tf.cast(self.D, tf.complex128)
#         a = k.dot(self.S, self.F)
#         self.results = k.dot(k.dot(self.S, self.F), self.D)
#         super(dictionary, self).build(input_shape)
#
#     def call(self, inputs, **kwargs):
#         inputs = tf.cast(inputs, tf.complex128)
#         return k.dot(inputs, self.results)
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1], self.D.shape[1])
#
#     def get_config(self):
#         config = {
#             'S_init': self.S_init,
#             'F_init': self.F_init,
#             'D_init': self.D_init,
#             'results': self.results
#         }
#         base_config = super(dictionary, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


def get_noise(signal, snr):
    # signal_power = (1 / int(signal.shape[1])) * k.sum(signal ** 2)
    signal_power = k.mean(signal ** 2, axis=1, keepdims=True)
    noise_var = signal_power / (10 ** (snr / 10))
    noise = k.random_normal(shape=k.shape(signal))
    noise = k.sqrt(noise_var) * noise
    return noise + signal


def matrix_H(inputs):
    return inputs.conj().T


def dftmtx(N):
    # return tf.fft(tf.eye(N, dtype='complex128'))
    return np.fft.fft(np.eye(N))


def dictionary_matrix(q):
    G_m = q * m
    G_n = q * n
    G_ms = N_ms

    theta_BS_vir_ang = np.arange(-1, 1, 2/G_m)
    phi_BS_vir_ang = np.arange(-1, 1, 2/G_n)
    theta_MS_vir_ang = np.arange(-1, 1, 2/G_ms)

    m_t = np.array(np.arange(0, m)).reshape(-1, 1)
    n_t = np.transpose(np.arange(0, n)).reshape(-1, 1)
    n_ms = np.transpose(np.arange(0, N_ms)).reshape(-1, 1)

    D_BS_1 = np.exp(-j * 2 * pi * d_ant * m_t * theta_BS_vir_ang) / np.sqrt(m)
    D_BS_2 = np.exp(-j * 2 * pi * d_ant * n_t * phi_BS_vir_ang) / np.sqrt(n)
    D_ms = np.exp(-j * 2 * pi * d_ant * n_ms * theta_MS_vir_ang) / np.sqrt(N_ms)

    D1 = np.kron(D_BS_1, D_BS_2).conjugate()

    D = np.kron(D1, D_ms)
    return D


# def add(inputs, mtx_init):
#     S = k.variable(value=mtx_init,
#                    dtype='complex128')
#     # S = tf.get_variable(name='S',
#     #                     initializer=mtx_init,
#     #                     trainable=False)
#     y = tf.cast(inputs, dtype='complex128')
#     y = k.reshape(y, (-1, output_dim))
#     a = k.dot(y, S)
#     out = k.reshape(a, shape=(-1, num, S.shape[1]))
#     return out


def change_dtype(inputs):
    return k.cast(inputs, dtype='float32')


def dic(inputs, P_real_init, P_imag_init):
    P_real = k.constant(value=P_real_init)
    P_imag = k.constant(value=P_imag_init)
    inputs_real = k.transpose(inputs[:, 0, :])
    inputs_imag = k.transpose(inputs[:, 1, :])
    output_real = Reshape((1, N_BS))(k.transpose(k.dot(P_real, inputs_real) - k.dot(P_imag, inputs_imag)))
    output_imag = Reshape((1, N_BS))(k.transpose(k.dot(P_real, inputs_imag) + k.dot(P_imag, inputs_real)))
    output = k.concatenate((output_real, output_imag), axis=1)
    return output


def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def residual_block_decoded(y):
    shortcut = y

    # y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    # y = add_common_layers(y)
    #
    # y = Conv2D(4, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    # y = add_common_layers(y)
    #
    y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(32, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(128, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    # y = Conv2D(512, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    # y = add_common_layers(y)
    #
    # y = Conv2D(1024, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    # y = add_common_layers(y)

    y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = BatchNormalization()(y)

    y = add([shortcut, y])
    y = LeakyReLU()(y)

    return y