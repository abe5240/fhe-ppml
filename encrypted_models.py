import torch.nn as nn

import tenseal as ts
from activations import *

class EncMLPNet:
    """Convert PyTorch MLP model into something capable of accepting ciphertext inputs."""

    def __init__(self, net, activation, order): 

        self.fc1_weight = net.linearL0_.weight.T.data.tolist()
        self.fc1_bias = net.linearL0_.bias.data.tolist()

        self.act0 = nn.ReLU()
        
        self.fc2_weight = net.linearL1_.weight.T.data.tolist()
        self.fc2_bias = net.linearL1_.bias.data.tolist()    


    def change_all_activations(self, new_activation):
        for layer in range(1):
            setattr(self, "act" + str(layer), new_activation)


    def forward(self, enc_x):

        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = self.act0(enc_x)
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias

        return enc_x
    

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class EncConvNet:
    """Convert PyTorch CNN model into something capable of accepting ciphertext inputs. The code for this can be found here: 
    https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb"""

    def __init__(self, net, activation, order):

        self.conv1_weight = net.convL0_.weight.data.view(net.convL0_.out_channels, net.convL0_.kernel_size[0], net.convL0_.kernel_size[1]).tolist()
        self.conv1_bias = net.convL0_.bias.data.tolist()
        self.act0 = nn.ReLU()
       
        self.fc1_weight = net.linearL1_.weight.T.data.tolist()
        self.fc1_bias = net.linearL1_.bias.data.tolist()
        self.act1 = nn.ReLU()

        self.fc2_weight = net.linearL2_.weight.T.data.tolist()
        self.fc2_bias = net.linearL2_.bias.data.tolist()


    def change_all_activations(self, new_activation):
        for layer in range(2):
            setattr(self, "act" + str(layer), new_activation)


    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        
        # Flatten encrypted values
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        enc_x = self.act0(enc_x)

        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = self.act1(enc_x)

        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias

        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
