import torch
import torch.nn as nn

from activations import *

class MLPNet(nn.Module):

    def __init__(self, num_classes=10):
        super(MLPNet, self).__init__()
        self.flatten   = nn.Flatten()
        self.linearL0_ = nn.Linear(in_features=784, out_features=256, bias=True) 
        self.bnL0_     = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act0      = nn.ReLU()

        self.linearL1_ = nn.Linear(in_features=256, out_features=num_classes, bias=True)

    def change_all_activations(self, new_activation):
        for layer in range(1):
            setattr(self, "act" + str(layer), new_activation)

    def forward(self, x):
        x = self.flatten(x)

        x = self.linearL0_(x)
        x = self.bnL0_(x)
        x = self.act0(x)

        x = self.linearL1_(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.convL0_    = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(2, 2))
        self.bnL0_      = nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act0       = nn.ReLU()

        self.linearL1_  = nn.Linear(in_features=864, out_features=120, bias=True)
        self.bnL1_      = nn.BatchNorm1d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1       = nn.ReLU()

        self.linearL2_  = nn.Linear(in_features=120, out_features=84, bias=True)
        self.bnL2_      = nn.BatchNorm1d(84, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2       = nn.ReLU()

        self.linearL3_  = nn.Linear(in_features=84, out_features=num_classes, bias=True)
        self.flatten    = nn.Flatten()

    def change_all_activations(self, new_activation):
        for layer in range(3):
            setattr(self, "act" + str(layer), new_activation)

    def forward(self, x):

        x = self.convL0_(x)
        x = self.bnL0_(x)
        x = self.act0(x)

        x = self.flatten(x)

        x = self.linearL1_(x)
        x = self.bnL1_(x)
        x = self.act1(x)

        x = self.linearL2_(x)
        x = self.bnL2_(x)
        x = self.act2(x)

        x = self.linearL3_(x)
        return x



if __name__ == "__main__":

    model = MLPNet()
    #model = LeNet()
    model.change_all_activations(leakyReluX(order=3))

    print(model)
