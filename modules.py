import numpy.linalg as alg

import numpy as np
import math
import torch

# import debugpy
# debugpy.debug_this_thread()


class NetF(torch.nn.Module):
    def __init__(self):
        super(NetF, self).__init__()

        self.net = torch.nn.Sequential(
            # #---------------------------
            torch.nn.Linear(2, 512, bias=False),
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(512, 512, bias=False),
            # torch.nn.ReLU(),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 2, bias=False),
            # torch.nn.Linear(2, 2, bias=False),
            # torch.nn.Tanh(),

        )

    def forward(self, x, dx):
        a = [x, dx]
        # a = Normalize(a)
        data_input = torch.cat(a, dim=1)
        # self.net[3].weight = torch.nn.Parameter(torch.Tensor(
        #     np.array([[1/20, 0], [0, 1/20]])))
        y = self.net(data_input)
        # y = mean_std(y)  # y*20 #
        y = y
        return y


