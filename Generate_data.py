import scipy.io
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

from torch.optim.lr_scheduler import LambdaLR


from sympy import *

from modules import NetF

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os

import copy

import debugpy
debugpy.debug_this_thread()


 
def net_predict_CLF(testdata, modelF):

    testdata = testdata.to(torch.float32)
    X = testdata[:, 0:2]
    X.requires_grad = True

    V_hat_ = modelF(X[:, 0:1],  X[:, 1:2]/12)
    V_hat = V_hat_ @  V_hat_.view(-1, 1)*1 + \
        X[:, 0:1]*X[:, 0:1]*1
    V_hat = V_hat*1

    grad_VX = torch.autograd.grad(
        outputs=V_hat, inputs=X, grad_outputs=torch.ones_like(V_hat), create_graph=True)
    dVdX = grad_VX[0].view(-1, 1, 2)*1

    return np.array(V_hat.detach()), dVdX.detach(), np.array(V_hat_.detach())





def systemGenerateData(x1_0, x2_0,  t_final, modelCLF):
 
    global invR1, invR2, u1b, u2b, uncertainty
    Qa = 9.35
    Qb = 0.41
    Qc = 0.41
    Qd = 0.02
    Q = np.array([[Qa, Qb], [Qc, Qd]])



    invR1 = 100
    invR2 = 500
    R = np.array([[1/invR1, 0], [0, 1/invR2]])

    if uncertainty == 0.9:
        CAs =   0.675056342111348 #0.999355860850161 # uncertainty=0.9
        Ts =   351.209701753034 #279.143142033298 # uncertainty=0.9
    elif uncertainty == 1:
        CAs = 0.5734
        Ts = 395.3268
    elif uncertainty == 0.95:
        CAs = 0.6271
        Ts = 373.0131

    elif uncertainty == 1.05:
        CAs = 0.5126
        Ts = 418.3428

    elif uncertainty == 0.85:
        CAs = 0.7182
        Ts = 329.8019
    elif uncertainty == 0.8:
        CAs = 0.7571
        Ts = 308.7153

    CF = 100e-3 * uncertainty
    CV = 0.1 * uncertainty
    Ck0 = 72e+9 * uncertainty
    CE = 8.314e+4 * uncertainty
    CR = 8.314 
    CT0 = 310 * uncertainty
    CDh = -4.78e+4 * uncertainty
    Ccp = 0.239 * uncertainty
    Crho = 1000 * uncertainty

    CA0s = 1
    Qs = 0
    lambda0 = 5000
    alpha0 = 0.1
    ua = np.array([[0.1], [0.1]])
    dx = np.array([[0.0], [0.0]])

    t_step = 0.001
    t_final = t_final

    sumV = 0

    x1, x2 = x1_0, x2_0

    x1_list = list()  # evolution of state over time
    x2_list = list()  # evolution of state time
    u1_list = list()
    u2_list = list()
    hat_V_list = list()
    dVdt_list = list()

    t_list = list()
    save_data = []
    NNoutput_list10 = []
    NNoutput_list20 = []
    dVdX1_list = []
    dVdX2_list = []

    lastV = 0
    dVdt = 0

    for i in range(int(t_final / t_step)):

        f1 = (CF / CV) * (- x1) - Ck0 * np.exp(-CE / (CR * (x2 + Ts))) * \
            (x1 + CAs)+(CF / CV) * (CA0s-CAs)
        f2 = (CF / CV) * (-x2) + (-CDh / (Crho * Ccp)) * Ck0 * \
            np.exp(-CE / (CR * (x2 + Ts))) * (x1 + CAs) + CF*(CT0-Ts)/CV

        g1 = CF / CV
        g2 = 1 / (Crho * Ccp * CV)

        f = np.array([[f1], [f2]])
        g = np.array([[g1, 0], [0, g2]])

        # -----------------------------------------------------------------
        # optimal sontag's controller
        # ---------------------------------------------------------------
        #-- barrier function---#

        test_data_V = torch.tensor(np.array([[(x1)/1, (x2)/1]]))

        predict_V = net_predict_CLF(test_data_V, modelCLF)

        V = predict_V[0]
        hat_VV = V[0][0]
        dVdX = predict_V[1].view(-1, 1)

       


        LfV = dVdX[0][0] * f1 + dVdX[1][0] * f2 

        LgV1 = g1 * dVdX[0][0] 
        LgV2 = g2 * dVdX[1][0] 

        QX = np.array([[x1, x2]])@Q@np.array([[x1], [x2]])

        if (abs(x1) < 0.1) and (abs(x2) < 1):
            deta_u = 0.001*(abs(x1)+abs(x2))  # 0.002*(abs(x1)+abs(x2))
        else:
            deta_u = 0.001  # 0.002
        deta_u =0
        #-- control input---#
        if (abs(LgV1) > 1e-5):
            kx1 = (LfV+math.sqrt(math.pow(LfV, 2)+invR1*math.pow(LgV1, 2)*(QX))) / \
                (math.pow(LgV1, 2)+deta_u)
            ub1 = -kx1*LgV1
        else:
            ub1 = 0

        if (abs(LgV2) > 1e-5):
            kx2 = (LfV+math.sqrt(math.pow(LfV, 2)+invR2*math.pow(LgV2, 2)*(QX))) / \
                (math.pow(LgV2, 2))  # +deta_u
            ub2 = -kx2*LgV2
        else:
            ub2 = 0

        ua_ = np.array([[ub1], [ub2]])

        ua[0] = ua_[0][0]
        ua[1] = ua_[1][0]



        if (ua[0] > u1b):  # 1
            ua[0] = u1b
        elif (ua[0] < -u1b):
            ua[0] = -u1b

        if ua[1] > u2b:  # 0.0167
            ua[1] = u2b
        elif ua[1] < -u2b:
            ua[1] = -u2b

        sumV = sumV + \
            np.array([[x1, x2]]) @ Q @ np.array([[x1], [x2]]) + \
            np.transpose(ua) @ R @ ua

        dx = f+g @ ua

        dVdt = np.transpose(dVdX)@dx
        dVdt = dVdt[0][0]

        U = np.transpose(ua) @ R @ ua

        # ---------------------------------
        # NN training data
        # ---------------------------------

        save_data0 = np.array(
            [[(x1)/1, (x2)/1, dx[0][0], dx[1][0], U[0][0]]])
        save_data.append(save_data0)
        traindata = np.array(save_data)
 

        # update
        dx = f+g @ ua
        x1 = x1+dx[0][0]*t_step
        x2 = x2+dx[1][0]*t_step

        # save data
        # if i % 10 == 0:
        x1_list.append(x1)
        x2_list.append(x2)
        u1_list.append(ua[0][0])
        u2_list.append(ua[1][0])


        dVdt_list.append(dVdt)
        dVdX1_list.append(dVdX[0][0])
        dVdX2_list.append(dVdX[1][0])
        t_list.append(i*t_step)

    return x1_list, x2_list, u1_list, u2_list, hat_V_list, t_list, dVdt_list, sumV, traindata


if __name__ == "__main__":

    invR1 = 1
    invR2 = 1

    final_C0 = 0.7
    flag_CLBF = 0

    uncertainty=0.9
    uncertainty0=1
    u1b = 5  # 2.2  # 1  # 15
    u2b = 0.167  # 4.67  # 0.0167  # 25

 

    modelCLF = NetF()

    modelCLF.load_state_dict(torch.load("saved_model_pre.pth"))

 
 
    x1_0 = -0.2  # unsafe
    x2_0 = 2
 

    t_final = 5
    traindata = systemGenerateData(x1_0, x2_0,  t_final, modelCLF)
    x1_list = traindata[0]
    x2_list = traindata[1]
    u1_list = traindata[2]
    u2_list = traindata[3]

    hat_V_list = traindata[4]
    t_list = traindata[5]
    dVdt_list = traindata[6]
    
    np.save(r"data_target.npy", traindata[8])





    plt.figure(1, figsize=(4, 4))
    plt.plot(t_list, u1_list, "r")
    plt.plot(t_list, u2_list, "b-.")
    plt.legend(labels=['u1', 'u2'])
    plt.grid()
    plt.ylabel('u1,u2')
    plt.xlabel('Time')
    



    plt.figure(6, figsize=(4, 4))
    plt.plot(t_list, x1_list, "r")
    plt.plot(t_list, x2_list, "b")
    plt.legend(labels=['x1', 'x2'])
    plt.grid()
    plt.ylabel('x1,x2')
    plt.xlabel('Time')

    plt.show()

    input()
