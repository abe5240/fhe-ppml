import torch
import torch.nn as nn
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


class leakyReluX(nn.Module):
    """Leaky ReLU polynomial approximation"""

    def __init__(self, order, a):
        super().__init__()
        self.order = order
        xs = np.arange(-a,a,1e-3)
        self.coeffs = np.flip(np.polyfit(xs, np.where(xs>0,xs,xs*0.1), order))

    def forward(self, x):
        if (self.order < 2) or (self.order > 10):
            raise SystemExit("Order can't be <2 or >10")
        
        out = 0 
        for i in range(len(self.coeffs)):
            out += self.coeffs[i] * x**i
        return out


class sigmoidX(nn.Module):
    """Sigmoid polynomial approximation"""

    def __init__(self, order, a):
        super().__init__()
        self.order = order
        xs = np.arange(-a,a,1e-3)
        self.coeffs = np.flip(np.polyfit(xs, 1/(1+np.exp(-xs)), order))

    def forward(self, x):
        if (self.order < 2) or (self.order > 10):
            raise SystemExit("Order can't be <2 or >10")
        
        out = 0 
        for i in range(len(self.coeffs)):
            out += self.coeffs[i] * x**i
        return out


class softplusX(nn.Module):
    """Softplus polynomial approximation"""

    def __init__(self, order, a):
        super().__init__()
        self.order = order
        xs = np.arange(-a,a,1e-3)
        self.coeffs = np.flip(np.polyfit(xs, np.log(1+np.exp(xs)), order))

    def forward(self, x):
        if (self.order < 2) or (self.order > 10):
            raise SystemExit("Order can't be <2 or >10")
        
        out = 0 
        for i in range(len(self.coeffs)):
            out += self.coeffs[i] * x**i
        return out


if __name__ == "__main__":
    
    a = 5
    xs = np.linspace(-a,a,1000)
    fig, axs = plt.subplots(3, 9, sharex=True, figsize=(16,8))

    for i in range(2,11):

        lrelu    = leakyReluX(order=i, a=a)
        sigmoid  = sigmoidX(order=i, a=a)
        softplus = softplusX(order=i, a=a)

        lrelu_out    = lrelu.forward(xs)
        sigmoid_out  = sigmoid.forward(xs)
        softplus_out = softplus.forward(xs)

        if i == 2:
            axs[0][i-2].plot(xs, np.where(xs>0,xs,xs*0.1), '--', color='gray', label='True', zorder=10)
            axs[0][i-2].plot(xs, lrelu_out, color='tab:blue', label='Poly.')

            axs[1][i-2].plot(xs, 1/(1+np.exp(-xs)), '--', color='gray', zorder=10)
            axs[1][i-2].plot(xs, sigmoid_out, color='tab:blue')

            axs[2][i-2].plot(xs,  np.log(1+np.exp(xs)), '--', color='gray', zorder=10)
            axs[2][i-2].plot(xs, softplus_out, color='tab:blue')         

        else:
            axs[0][i-2].plot(xs, np.where(xs>0,xs,xs*0.1), '--', color='gray', zorder=10)
            axs[1][i-2].plot(xs, 1/(1+np.exp(-xs)), '--', color='gray', zorder=10)
            axs[2][i-2].plot(xs, np.log(1+np.exp(xs)), '--', color='gray', zorder=10)

            axs[0][i-2].plot(xs, lrelu_out, color='tab:blue')
            axs[1][i-2].plot(xs, sigmoid_out, color='tab:blue')
            axs[2][i-2].plot(xs, softplus_out, color='tab:blue')

        axs[0][i-2].set_title(f"Order {i}")

        axs[0][i-2].set_ylim([-5, 5])
        axs[1][i-2].set_ylim([-2, 2])
        axs[2][i-2].set_ylim([-5, 5])

        axs[0][0].set_ylabel("Leaky ReLU")
        axs[1][0].set_ylabel("Sigmoid")
        axs[2][0].set_ylabel("Softplus")

        if i != 2:
            axs[0][i-2].set_yticks([])
            axs[1][i-2].set_yticks([])
            axs[2][i-2].set_yticks([])

    axs[0][0].legend()
    #plt.tight_layout()
    plt.suptitle("Polynomial Approximations of Activation Functions")
    plt.show()

    






