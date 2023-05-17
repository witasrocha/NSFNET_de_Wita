import pyJHTDB
import pyJHTDB.dbinfo
import numpy as np
import matplotlib.pyplot as plt
import time


def get_data(point_coords, time):
    """
    Get velocity and pressure at specified spatial points and a specified time in channel flow database.
    :param point_coords: Spatial coordinates of the data points of interest. Must be in single precision.
    :param time: Time of interest.
    :return: Velocity and velocity gradient arrays.
    """

    # Create library object
    lJHTDB = pyJHTDB.libJHTDB()

    # Initialize library object
    lJHTDB.initialize()

    # Get velocity
    u = lJHTDB.getData(time, point_coords,
                       sinterp='Lag4',
                       data_set='channel',
                       getFunction='getVelocity')

    # Get velocity gradient
    p = lJHTDB.getData(time, point_coords,
                            sinterp='Lag4',
                            data_set='channel',
                            getFunction='getPressure')

    # Finalize library object
    lJHTDB.finalize()

    return u, p


xnode = np.linspace(12.47,12.66,191)
ynode = np.linspace(-0.9,-0.7,201)
znode = np.linspace(4.61,4.82,211)
total_times = np.array(list(range(4000)), dtype = np.float32) * 0.0065
frames1 = np.arange(81)
frames2 = np.arange(129)

points = np.zeros((191, 201, 211, 3), np.float32)
points[:, :, :, 0] = xnode[:, None, None]
points[:, :, :, 1] = ynode[None, :, None]
points[:, :, :, 2] = znode[None, None, :]


points1 = points.reshape(-1,3)

idx_ini = np.random.choice(points1.shape[0], 33524, replace = False)

train_ini1 = points1[idx_ini, :]
train_iniv1 = np.zeros((33524, 3), np.float32)
train_inip1 = np.zeros((33524, 1), np.float32)

size = int(33524 / 17)
for i in range(17):
    start = time.time()  # start timer

    train_iniv1[0 + size * i: size + size * i, :], train_inip1[0 + size * i: size + size * i, :] = get_data(
        train_ini1[0 + size * i: size + size * i, :], 0)

    elapsed = time.time() - start  # end timer
    print('Elapsed time = %.2f seconds' % elapsed)

print(train_iniv1.shape, train_inip1.shape)


np.save('train_ini1.npy',train_ini1)
np.save('train_iniv1.npy',train_iniv1)
np.save('train_inip1.npy',train_inip1)

points2 = points1[:, :][points1[:,0] == 12.47]
points3 = points1[:, :][points1[:,0] == 12.66]
points4 = points1[:, :][points1[:,1] == -0.9]
points5 = points1[:, :][points1[:,1] == -0.7]
points6 = points1[:, :][points1[:,2] == 4.61]
points7 = points1[:, :][points1[:,2] == 4.82]
print(points2.shape, points3.shape, points4.shape, points5.shape, points6.shape, points7.shape)

train_b1 = np.concatenate([points2, points3, points4, points5, points6, points7],0)
train_b1.shape

idxb = np.random.choice(train_b1.shape[0], 6644, replace=False)

train_b2 = train_b1[idxb,:]
train_b2.shape

train_xb1 = np.zeros((6644*129, 4), np.float32)
train_vb1 = np.zeros((6644*129, 3), np.float32)
train_pb1 = np.zeros((6644*129, 1), np.float32)

frames = np.arange(129)
# Get data
for frame in frames:
    print(frame)
    for i in range(2):
        train_vb1[6644*frame + 3322*i: 3322+6644*frame+3322*i, :], train_pb1[6644*frame + 3322*i: 3322+6644*frame+3322*i, :] = get_data(train_b2[3322*i: 3322+3322*i, :], t)
    train_xb1[6644*frame: 6644+6644*frame, 0:3] = train_b2
    train_xb1[6644*frame: 6644+6644*frame, 3] = t

np.save('train_xb1.npy',train_xb1)
np.save('train_vb1.npy',train_vb1)
np.save('train_pb1.npy',train_pb1)


x0_train = train_ini1[:,0:1]
y0_train = train_ini1[:,1:2]
z0_train = train_ini1[:,2:3]
t0_train = np.zeros(train_ini1[:,0:1].shape, np.float32)
u0_train = train_iniv1[:,0:1]
v0_train = train_iniv1[:,1:2]
w0_train = train_iniv1[:,2:3]

xb_train = train_xb1[:,0:1]
yb_train = train_xb1[:,1:2]
zb_train = train_xb1[:,2:3]
tb_train = train_xb1[:,3:4]
ub_train = train_vb1[:,0:1]
vb_train = train_vb1[:,1:2]
wb_train = train_vb1[:,2:3]

x_train1 = xnode.reshape(-1,1)[np.random.choice(191, 20000, replace=True),:]
y_train1 = ynode.reshape(-1,1)[np.random.choice(201, 20000, replace=True),:]
z_train1 = znode.reshape(-1,1)[np.random.choice(211, 20000, replace=True),:]
x_train = np.tile(x_train1,(129, 1))
y_train = np.tile(y_train1,(129, 1))
z_train = np.tile(z_train1,(129, 1))

total_times1 = np.array(list(range(129))) * 0.0065
t_train1 = total_times1.repeat(20000)
t_train = t_train1.reshape(-1,1)

print(x0_train.shape,y0_train.shape,
      z0_train.shape,t0_train.shape,
      u0_train.shape,v0_train.shape,
      w0_train.shape,
      xb_train.shape,yb_train.shape,
      zb_train.shape,tb_train.shape,
      ub_train.shape,vb_train.shape,
      wb_train.shape,
      x_train.shape, y_train.shape,
      z_train.shape, t_train.shape)

idx_test = np.random.choice(points1.shape[0],1000,replace = False)
test_l = points1[idx_test, :] # test points locations
#choose several different time step

test_v1, test_p1 = get_data(test_l, 0.0065)
test_v2, test_p2 = get_data(test_l, 31 * 0.0065)
test_v3, test_p3 = get_data(test_l, 61 * 0.0065)
test_v4, test_p4 = get_data(test_l, 91 * 0.0065)
test_v5, test_p5 = get_data(test_l, 121 * 0.0065)

test_v = np.concatenate((test_v1, test_v2, test_v3, test_v4, test_v5), 0)
test_p = np.concatenate((test_p1, test_p2, test_p3, test_p4, test_p5), 0)

test42 = np.concatenate((test_v, test_p), 1)

np.save('test42_l.npy', test_l)
np.save('test42_vp.npy', test42)