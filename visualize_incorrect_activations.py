import torch
import torch.nn as nn
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from activations import *

if __name__ == "__main__":
    
    a = 5
    xs = np.linspace(-10,10,1000)
    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(5,5))

    orders = [2,6,10]
    for i in range(len(orders)):

        lrelu    = leakyReluX(order=orders[i], a=a)
        sigmoid  = sigmoidX(order=orders[i], a=a)
        softplus = softplusX(order=orders[i], a=a)

        lrelu_out    = lrelu.forward(xs)
        sigmoid_out  = sigmoid.forward(xs)
        softplus_out = softplus.forward(xs)

        if i == 2:
            axs[0][i].plot(xs, np.where(xs>0,xs,xs*0.1), '--', color='tab:red', zorder=1, linewidth=2)
            axs[0][i].plot(xs, lrelu_out, color='tab:blue')

            axs[1][i].plot(xs, 1/(1+np.exp(-xs)), '--', color='tab:red', zorder=1, linewidth=2)
            axs[1][i].plot(xs, sigmoid_out, color='tab:blue')

            axs[2][i].plot(xs,  np.log(1+np.exp(xs)), '--', color='tab:red', zorder=1, linewidth=2, label='True')
            axs[2][i].plot(xs, softplus_out, color='tab:blue', label='Poly.')         

        else:
            axs[0][i].plot(xs, np.where(xs>0,xs,xs*0.1), '--', color='tab:red', zorder=1, linewidth=2)
            axs[1][i].plot(xs, 1/(1+np.exp(-xs)), '--', color='tab:red', zorder=1, linewidth=2)
            axs[2][i].plot(xs, np.log(1+np.exp(xs)), '--', color='tab:red', zorder=1, linewidth=2)

            axs[0][i].plot(xs, lrelu_out, color='tab:blue')
            axs[1][i].plot(xs, sigmoid_out, color='tab:blue')
            axs[2][i].plot(xs, softplus_out, color='tab:blue')

        axs[0][i].set_title(f"Order {orders[i]}")

        axs[0][i].set_ylim([-5, 5])
        axs[1][i].set_ylim([-2, 2])
        axs[2][i].set_ylim([-5, 5])

        axs[0][0].set_ylabel("Leaky ReLU")
        axs[1][0].set_ylabel("Sigmoid")
        axs[2][0].set_ylabel("Softplus")

        if i != 0:
            axs[0][i].set_yticks([])
            axs[1][i].set_yticks([])
            axs[2][i].set_yticks([])

    axs[2][2].legend()
    #axs[2][0].set_xlabel("x")
    axs[2][1].set_xlabel("x")
    plt.tight_layout()
    #plt.suptitle("Polynomial Approximations of Activation Functions")
    plt.savefig("incorrect_polys.eps")
    plt.show()

    







