import numpy as np
import pandas as pd
import os



def fitfunc(xi, coort, sim1, ind1, sim2, ind2):
    w1k = sim1[:xi[0].astype(int) + 1]
    w2k = sim2[:xi[0].astype(int) + 1]
    w1 = w1k / np.sum(w1k)
    w2 = w2k / np.sum(w2k)
    locp1 = np.sum(np.tile(w1, (1, 2)) * coort[np.sum(ind1[:xi[0].astype(int) + 1],1),:], 0)
    locp2 = np.sum(np.tile(w2, (1, 2)) * coort[np.sum(ind1[:xi[0].astype(int) + 1],1),:], 0)
    fit_val = np.sqrt(np.sum(np.square(locp1 - xi[1:]))) + np.sqrt(np.sum(np.square(locp2 - xi[1:])))
    return fit_val



if __name__ == '__main__':
    TrainFile = r'train_mean.csv'
    TestFile = r'test_mean.csv'
    CoordFileTr = r'coordinates_tr.csv'
    CoordFileTe = r'coordinates_te.csv'

    pwd = os.getcwd()
    os.chdir(os.path.dirname(TrainFile))
    train = pd.read_csv(os.path.basename(TrainFile))
    os.chdir(pwd)
    pwd = os.getcwd()
    os.chdir(os.path.dirname(TestFile))
    test = pd.read_csv(os.path.basename(TestFile))
    os.chdir(pwd)
    pwd = os.getcwd()
    os.chdir(os.path.dirname(CoordFileTr))
    coor_tr = pd.read_csv(os.path.basename(CoordFileTr))
    os.chdir(pwd)
    pwd = os.getcwd()
    os.chdir(os.path.dirname(CoordFileTe))
    coor_te = pd.read_csv(os.path.basename(CoordFileTe))
    os.chdir(pwd)
    
    rss_t = train.values
    rss_e = test.values
    coor_t = coor_tr.values
    coor_e = coor_te.values
    N_t = rss_t.shape[0]
    N_e = rss_e.shape[0]
    M_AP = rss_t.shape[1]
    
    coor_p = np.zeros((N_e, 2))
    dis_1 = np.zeros((N_t, 1))
    dis_2 = np.zeros((N_t, 1))
    
    for test in range(N_e):
        print('test = ' + str(test))
        rss_ei = rss_e[test]
        for i in range(N_t):
            dis_1[i] = np.sqrt(np.sum(np.square(rss_ei - rss_t[i])))
            dis_2[i] = 1 - (np.sum(rss_ei * rss_t[i])) / (np.sqrt(np.sum(np.square(rss_ei))) * np.sqrt(np.sum(np.square(rss_t[i]))))
        dis_1s = np.sort(dis_1, 0)
        dis_2s = np.sort(dis_2, 0)
        ind_1 = np.argsort(dis_1, 0)
        ind_2 = np.argsort(dis_2, 0)
        sim_1 = np.power(10, -dis_1s)
        sim_2 = np.power(10, -dis_2s)

        N = 100
        D = 3
        w_min = 0.4
        w_max = 0.9
        c1 = c2 = 1.49
        T_max = 10000

        BL = np.array([2, 0, 0])
        BU = np.array([8, width,  lenth])
        BLv = 0.01 * BL
        BUv = 0.01 * BU

        x_min = np.tile(BL, (N, 1))
        x_max = np.tile(BU, (N, 1))
        v_min = np.tile(BLv, (N, 1))
        v_max = np.tile(BUv, (N, 1))
        x = x_min + (x_max - x_min) * np.random.rand(N,D)
        v = v_min + (v_max - v_min) * np.random.rand(N,D)
        x[:, 0] = np.round(x[:, 0])

        fit = np.zeros(N)
        for i in range(N):
            fit[i] = fitfunc(x[i, :], coor_t, sim_1, ind_1, sim_2, ind_2)
        pbest_val = fit
        pbest_pos = x
        gbest_val = np.min(pbest_val)
        gbest_ind = np.argmin(fit)
        gbest_pos = x[gbest_ind,:]

        t = 0
        while t < T_max:
            t += 1
            w = w_min + (w_max - w_min) * (T_max - t) / T_max
            v = w * v + c1 * np.random.rand(N,D) * (pbest_pos - x) + c2 * np.random.rand(N,D) * (gbest_pos - x)
            for i in range(N):
                for j in range(D):
                    if v[i, j] < v_min[i, j]:
                        v[i, j] = v_min[i, j]
            for i in range(N):
                for j in range(D):
                    if v[i, j] > v_max[i, j]:
                        v[i, j] = v_max[i, j]
            x += v
            x[:, 0] = np.round(x[:, 0])
            for i in range(N):
                for j in range(D):
                    if x[i, j] < x_min[i, j]:
                        x[i, j] = x_min[i, j]
            for i in range(N):
                for j in range(D):
                    if x[i, j] > x_max[i, j]:
                        x[i, j] = x_max[i, j]
            for i in range(N):
                fit[i] = fitfunc(x[i, :], coor_t, sim_1, ind_1, sim_2, ind_2)
                if fit[i] < pbest_val[i]:
                    pbest_val[i] = fit[i]
                    pbest_pos[i] = x[i,:]
            gbest_val_if = np.min(pbest_val)
            gbest_ind_if = np.argmin(pbest_val)
            if gbest_val_if < gbest_val:
                gbest_val = gbest_val_if
                gbest_pos = x[gbest_ind_if,:]
        coor_p[test, :] = gbest_pos[1:]
