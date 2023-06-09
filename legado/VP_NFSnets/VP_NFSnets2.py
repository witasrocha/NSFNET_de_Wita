import tensorflow as tf
import numpy as np
import scipy.io
import time

# set random seed
np.random.seed(1234)
tf.set_random_seed(1234)

# #################################################
# ###############plotting function#################
# #################################################
#
# def figsize(scale, nplots = 1):
#     fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
#     inches_per_pt = 1.0/72.27                       # Convert pt to inch
#     golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
#     fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
#     fig_height = nplots*fig_width*golden_mean              # height in inches
#     fig_size = [fig_width,fig_height]
#     return fig_size
#
# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": "DejaVu Sans",
#     "font.sans-serif": [],                   # blank entries should cause plots to inherit fonts from the document
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "axes.labelsize": 10,               # LaTeX default is 10pt font.
#     "font.size": 10,
#     "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }
#
# # set mpl parameters
#
# mpl.rcParams.update(pgf_with_latex)
#
# import matplotlib.pyplot as plt
#
# # I make my own newfig and savefig functions
#
# def newfig(width, nplots = 1):
#     fig = plt.figure(figsize=figsize(width, nplots))
#     ax = fig.add_subplot(111)
#     return fig, ax
#
# def savefig(filename, crop = True):
#     if crop == True:
# #        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
#         plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
# #         plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
#     else:
# #        plt.savefig('{}.pgf'.format(filename))
#         plt.savefig('{}.pdf'.format(filename))
# #         plt.savefig('{}.eps'.format(filename))

#############################################
###################VP NSFnet#################
#############################################

class VPNSFnet:
    # Initialize the class
    def __init__(self, x0, y0, t0, u0, v0, xb, yb, tb, ub, vb, x, y, t, layers):
        X0 = np.concatenate([x0, y0, t0], 1)  # remove the second bracket
        Xb = np.concatenate([xb, yb, tb], 1)
        X = np.concatenate([x, y, t], 1)

        self.lowb = X.min(0)  # minimal number in each column
        self.upb = X.max(0)

        self.X0 = X0
        self.Xb = Xb
        self.X = X

        self.x0 = X0[:, 0:1]
        self.y0 = X0[:, 1:2]
        self.t0 = X0[:, 2:3]
        self.xb = Xb[:, 0:1]
        self.yb = Xb[:, 1:2]
        self.tb = Xb[:, 2:3]
        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]

        self.u0 = u0
        self.v0 = v0
        self.ub = ub
        self.vb = vb

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_ini_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.y_ini_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])
        self.t_ini_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u_ini_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v_ini_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])

        self.x_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.y_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.t_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.tb.shape[1]])
        self.u_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.v_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.vb.shape[1]])

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_ini_pred, self.v_ini_pred, self.p_ini_pred = self.net_NS(self.x_ini_tf, self.y_ini_tf, self.t_ini_tf)

        self.u_boundary_pred, self.v_boundary_pred, self.p_boundary_pred = \
            self.net_NS(self.x_boundary_tf, self.y_boundary_tf, self.t_boundary_tf)

        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_e_pred = \
            self.net_f_NS(self.x_tf, self.y_tf, self.t_tf)

        alpha, beta = 1, 1

        # set loss function
        self.loss = alpha * tf.reduce_mean(tf.square(self.u_ini_tf - self.u_ini_pred)) + \
                    alpha * tf.reduce_mean(tf.square(self.v_ini_tf - self.v_ini_pred)) + \
                    beta * tf.reduce_mean(tf.square(self.u_boundary_tf - self.u_boundary_pred)) + \
                    beta * tf.reduce_mean(tf.square(self.v_boundary_tf - self.v_boundary_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_e_pred))

        # set optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 100 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

# initialize the weight and bias
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

# xavier used to initialize the weight
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

# generate the neural network
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

# ###################without assume###############
    # supervised data driven
    def net_NS(self, x, y, t):

        u_v_p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        return u, v, p

    # unsupervised NS residual
    def net_f_NS(self, x, y, t):

        u_v_p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = u_t + (u * u_x + v * u_y) + p_x - 0.01 * (u_xx + u_yy)
        f_v = v_t + (u * v_x + v * v_y) + p_y - 0.01 * (v_xx + v_yy)
        f_e = u_x + v_y

        return u, v, p, f_u, f_v, f_e

    def callback(self, loss):
        print('Loss: %.3e' % loss)

# tf_dict is used to connect model variable and data

    def Adam_train(self, nIter=5000, learning_rate=1e-3):

        tf_dict = {self.x_ini_tf: self.x0, self.y_ini_tf: self.y0,
                   self.t_ini_tf: self.t0, self.u_ini_tf: self.u0,
                   self.v_ini_tf: self.v0, self.x_boundary_tf: self.xb,
                   self.y_boundary_tf: self.yb, self.t_boundary_tf: self.tb,
                   self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.learning_rate: learning_rate}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
# later here should add dynamic strategy here
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

# two step train BFGS used to finetune the result
    def BFGS_train(self):

        tf_dict = {self.x_ini_tf: self.x0, self.y_ini_tf: self.y0,
                   self.t_ini_tf: self.t0, self.u_ini_tf: self.u0,
                   self.v_ini_tf: self.v0, self.x_boundary_tf: self.xb,
                   self.y_boundary_tf: self.yb, self.t_boundary_tf: self.tb,
                   self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

# 不需要改变 可能需要注意x_tf等

    def predict(self, x_star, y_star, t_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, p_star

if __name__ == "__main__":
    # when model is directly run this will implement
    # supervised

    N_train = 40000

    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 3]

    # Load Data
    data = scipy.io.loadmat('C:/Users/zihao/PycharmProjects/untitled/PINNs-master/main/Data/cylinder_nektar_wake.mat')
    ###############################
    U_star = data['U_star']  # N x 2 x T
    # velocity nx2xt
    P_star = data['p_star']  # N x T
    # pressure nxt
    t_star = data['t']  # T x 1
    # time t
    X_star = data['X_star']  # N x 2
    # location nx2
    # n number of datapoint, t time steps
    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    # 横向复制 n行1列复制为n行t列
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    # 横向复制 之后转置

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1

    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data_domain = data1[:, :][data1[:, 2] <= 7]

    data_t0 = data_domain[:, :][data_domain[:, 2] == 0]

    data_y1 = data_domain[:, :][data_domain[:, 0] == 1]
    data_y8 = data_domain[:, :][data_domain[:, 0] == 8]
    data_x = data_domain[:, :][data_domain[:, 1] == -2]
    data_x2 = data_domain[:, :][data_domain[:, 1] == 2]

    data_sup_b_train = np.concatenate([data_y1, data_y8, data_x, data_x2], 0)

    idx = np.random.choice(data_domain.shape[0], N_train, replace=False)

    x_train = data_domain[idx, 0].reshape(-1, 1)
    y_train = data_domain[idx, 1].reshape(-1, 1)
    t_train = data_domain[idx, 2].reshape(-1, 1)

    x0_train = data_t0[:, 0].reshape(-1, 1)
    y0_train = data_t0[:, 1].reshape(-1, 1)
    t0_train = data_t0[:, 2].reshape(-1, 1)
    u0_train = data_t0[:, 3].reshape(-1, 1)
    v0_train = data_t0[:, 4].reshape(-1, 1)

    xb_train = data_sup_b_train[:, 0].reshape(-1, 1)
    yb_train = data_sup_b_train[:, 1].reshape(-1, 1)
    tb_train = data_sup_b_train[:, 2].reshape(-1, 1)
    ub_train = data_sup_b_train[:, 3].reshape(-1, 1)
    vb_train = data_sup_b_train[:, 4].reshape(-1, 1)

    model = VPNSFnet(x0_train, y0_train, t0_train, u0_train, v0_train, xb_train,
                     yb_train, tb_train, ub_train, vb_train, x_train, y_train, t_train,
                     layers)

    model.Adam_train(1000, 1e-2)
    model.Adam_train(10000, 1e-3)
    model.Adam_train(50000, 1e-4)
    model.Adam_train(50000, 1e-5)
    model.BFGS_train()

    # Test model performance at 1 second
    snap1 = np.array([10])
    x_star1 = X_star[:, 0:1]
    y_star1 = X_star[:, 1:2]
    t_star1 = TT[:, snap1]

    u_star1 = U_star[:, 0, snap1]
    v_star1 = U_star[:, 1, snap1]
    p_star1 = P_star[:, snap1]

    # Prediction
    u_pred1, v_pred1, p_pred1 = model.predict(x_star1, y_star1, t_star1)

    # Error
    error_u1 = np.linalg.norm(u_star1 - u_pred1, 2) / np.linalg.norm(u_star1, 2)
    error_v1 = np.linalg.norm(v_star1 - v_pred1, 2) / np.linalg.norm(v_star1, 2)
    error_p1 = np.linalg.norm(p_star1 - p_pred1, 2) / np.linalg.norm(p_star1, 2)

    print('performance at 1 second:\n')
    print('Error u1: %e' % error_u1)
    print('Error v1: %e' % error_v1)
    print('Error p1: %e' % error_p1)

    # Test model performance at 2 second
    snap2 = np.array([20])
    x_star2 = X_star[:, 0:1]
    y_star2 = X_star[:, 1:2]
    t_star2 = TT[:, snap2]

    u_star2 = U_star[:, 0, snap2]
    v_star2 = U_star[:, 1, snap2]
    p_star2 = P_star[:, snap2]

    # Prediction
    u_pred2, v_pred2, p_pred2 = model.predict(x_star2, y_star2, t_star2)

    # Error
    error_u2 = np.linalg.norm(u_star2 - u_pred2, 2) / np.linalg.norm(u_star2, 2)
    error_v2 = np.linalg.norm(v_star2 - v_pred2, 2) / np.linalg.norm(v_star2, 2)
    error_p2 = np.linalg.norm(p_star2 - p_pred2, 2) / np.linalg.norm(p_star2, 2)

    print('performance at 2 second:\n')
    print('Error u2: %e' % error_u2)
    print('Error v2: %e' % error_v2)
    print('Error p2: %e' % error_p2)

    # Test model performance at 3 second
    snap3 = np.array([30])
    x_star3 = X_star[:, 0:1]
    y_star3 = X_star[:, 1:2]
    t_star3 = TT[:, snap3]

    u_star3 = U_star[:, 0, snap3]
    v_star3 = U_star[:, 1, snap3]
    p_star3 = P_star[:, snap3]

    # Prediction
    u_pred3, v_pred3, p_pred3 = model.predict(x_star3, y_star3, t_star3)

    # Error
    error_u3 = np.linalg.norm(u_star3 - u_pred3, 2) / np.linalg.norm(u_star3, 2)
    error_v3 = np.linalg.norm(v_star3 - v_pred3, 2) / np.linalg.norm(v_star3, 2)
    error_p3 = np.linalg.norm(p_star3 - p_pred3, 2) / np.linalg.norm(p_star3, 2)

    print('performance at 3 second:\n')
    print('Error u3: %e' % error_u3)
    print('Error v3: %e' % error_v3)
    print('Error p3: %e' % error_p3)

    # Test model performance at 6 second
    snap6 = np.array([60])
    x_star6 = X_star[:, 0:1]
    y_star6 = X_star[:, 1:2]
    t_star6 = TT[:, snap6]

    u_star6 = U_star[:, 0, snap6]
    v_star6 = U_star[:, 1, snap6]
    p_star6 = P_star[:, snap6]

    # Prediction
    u_pred6, v_pred6, p_pred6 = model.predict(x_star6, y_star6, t_star6)

    # Error
    error_u6 = np.linalg.norm(u_star6 - u_pred6, 2) / np.linalg.norm(u_star6, 2)
    error_v6 = np.linalg.norm(v_star6 - v_pred6, 2) / np.linalg.norm(v_star6, 2)
    error_p6 = np.linalg.norm(p_star6 - p_pred6, 2) / np.linalg.norm(p_star6, 2)

    print('performance at 6 second:\n')
    print('Error u6: %e' % error_u6)
    print('Error v6: %e' % error_v6)
    print('Error p6: %e' % error_p6)
