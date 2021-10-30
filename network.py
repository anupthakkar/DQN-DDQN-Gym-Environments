import numpy as np
import matplotlib.pyplot as plt
from operator import add
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # this is the basic architecture of the neural network.
        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, 200)
        self.layer3 = nn.Linear(200, 400)
        self.layer4 = nn.Linear(400, output_size)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        
    def forward(self, x):
        # we are using a simple Relu activation between the different layers.
        # print(x.shape)
        x = F.relu(self.layer1(x))
        # print(x.shape)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # print(x.shape)
        output = self.layer4(x)
        return output
        
    def save_model(self,filename='models/temporary_model.pth'):
        torch.save(self.state_dict(), filename)

    def load_model(self,filename='models/temporary_model.pth'):
        self.load_state_dict(torch.load(filename))

    def transfer_weights(self,model):
        self.load_state_dict(model.state_dict())