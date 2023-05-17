# author Zihao Hu
# time 5/12/2020

import sys

sys.path.append('PINNs-master/Utilities')

import tensorflow as tf
import numpy as np
import time
from NSFNET_de_Wita import VPNSFnet

# set random seed
np.random.seed(1234)
tf.set_random_seed(1234)

#############################################
###################VP NSFnet#################
#############################################


if __name__ == "__main__":
    # when model is directly run this will implement
    # supervised

    N_train = 10000

    layers = [4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]

    # Load Data
    train_ini1 = np.load('datasets/train_ini2.npy')
    train_iniv1 = np.load('datasets/train_iniv2.npy')
    train_inip1 = np.load('datasets/train_inip2.npy')
    train_xb1 = np.load('datasets/train_xb2.npy')
    train_vb1 = np.load('datasets/train_vb2.npy')
    train_pb1 = np.load('datasets/train_pb2.npy')

    xnode = np.linspace(12.47, 12.66, 191)
    ynode = np.linspace(-1, -0.0031, 998)
    znode = np.linspace(4.61, 4.82, 211)

    x0_train = train_ini1[:, 0:1]
    y0_train = train_ini1[:, 1:2]
    z0_train = train_ini1[:, 2:3]
    t0_train = np.zeros(train_ini1[:, 0:1].shape, np.float32)
    u0_train = train_iniv1[:, 0:1]
    v0_train = train_iniv1[:, 1:2]
    w0_train = train_iniv1[:, 2:3]

    xb_train = train_xb1[:, 0:1]
    yb_train = train_xb1[:, 1:2]
    zb_train = train_xb1[:, 2:3]
    tb_train = train_xb1[:, 3:4]
    ub_train = train_vb1[:, 0:1]
    vb_train = train_vb1[:, 1:2]
    wb_train = train_vb1[:, 2:3]

    x_train1 = xnode.reshape(-1, 1)[np.random.choice(191, 100000, replace=True), :]
    y_train1 = ynode.reshape(-1, 1)[np.random.choice(998, 100000, replace=True), :]
    z_train1 = znode.reshape(-1, 1)[np.random.choice(211, 100000, replace=True), :]
    x_train = np.tile(x_train1, (17, 1))
    y_train = np.tile(y_train1, (17, 1))
    z_train = np.tile(z_train1, (17, 1))

    total_times1 = np.array(list(range(17))) * 0.0065
    t_train1 = total_times1.repeat(100000)
    t_train = t_train1.reshape(-1, 1)

    model = VPNSFnet.VPNSFnet(x0_train, y0_train, z0_train, t0_train,
                     u0_train, v0_train, w0_train,
                     xb_train, yb_train, zb_train, tb_train,
                     ub_train, vb_train, wb_train,
                     x_train, y_train, z_train, t_train, layers)

    model.train(250, 150, 1e-3)
    model.train(4250, 150, 1e-4)
    model.train(500, 150, 1e-5)
    model.train(500, 150, 1e-6)

    # # Test Data
    # x_star = (np.random.rand(100, 1) - 1 / 2) * 2
    # y_star = (np.random.rand(100, 1) - 1 / 2) * 2
    # z_star = (np.random.rand(100, 1) - 1 / 2) * 2
    # t_star = np.random.randint(11, size=(100, 1)) / 10
    #
    # u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)
    #
    # # Prediction
    # u_pred, v_pred, w_pred, p_pred = model.predict(x_star, y_star, z_star, t_star)
    #
    # # Error
    # error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    # error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    # error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    # error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
    #
    # print('Error u: %e' % (error_u))
    # print('Error v: %e' % (error_v))
    # print('Error v: %e' % (error_w))
    # print('Error p: %e' % (error_p))
