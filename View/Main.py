import Controller.App as app
import Model.CarregaBases as cb

if __name__ == '__main__':
      app.N_train = 1000
      app.layers = [4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]

      xnode = cb.np.linspace(12.47, 12.66, 191)
      ynode = cb.np.linspace(-1, -0.0031, 998)
      znode = cb.np.linspace(4.61, 4.82, 211)

      train_ini1 = cb.train_ini1
      train_inip1 = cb.train_inip1
      train_iniv1 = cb.train_iniv1
      train_pb1=cb.train_pb1
      train_vb1 = cb.train_vb1
      train_xb1 = cb.train_xb1



      x0_train = train_ini1[:, 0:1]
      y0_train = train_ini1[:, 1:2]
      z0_train = train_ini1[:, 2:3]
      t0_train = cb.np.zeros(train_ini1[:, 0:1].shape, cb.np.float32)
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

      x_train1 = xnode.reshape(-1, 1)[cb.np.random.choice(191, 100000, replace=True), :]
      y_train1 = ynode.reshape(-1, 1)[cb.np.random.choice(998, 100000, replace=True), :]
      z_train1 = znode.reshape(-1, 1)[cb.np.random.choice(211, 100000, replace=True), :]
      x_train = cb.np.tile(x_train1, (17, 1))
      y_train = cb.np.tile(y_train1, (17, 1))
      z_train = cb.np.tile(z_train1, (17, 1))

      total_times1 = cb.np.array(list(range(17))) * 0.0065
      t_train1 = total_times1.repeat(100000)
      t_train = t_train1.reshape(-1, 1)

      print(t_train)