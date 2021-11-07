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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer1 = nn.Linear(input_size, 128).to(self.device)
        # self.bn1 = nn.BatchNorm1d(num_features=200)
        self.layer2 = nn.Linear(128, 128).to(self.device)
        self.layer3 = nn.Linear(128, 128).to(self.device)
        self.layer5 = nn.Linear(128, 128).to(self.device)
        self.layer6 = nn.Linear(128, 128).to(self.device)
        self.layer4 = nn.Linear(128, output_size).to(self.device)
        self.optimizer = optim.Adam(self.parameters())
        
        
    def forward(self, x):
        # we are using a simple Relu activation between the different layers.
        # print(x.shape)
        # print('entered forward')
        x = x.to(self.device)
        # print(x)
        # x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.layer1(x))
        # print(x)
        x = F.relu(self.layer2(x))
        # print(x)
        x = F.relu(self.layer3(x))
        # print(x)
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        output = self.layer4(x)
        # print(output)
        return output
        
    def save_model(self,filename='models/temporary_model.pth'):
        torch.save(self.state_dict(), filename)

    def load_model(self,filename='models/temporary_model.pth'):
        self.load_state_dict(torch.load(filename))

    def transfer_weights(self,model):
        self.load_state_dict(model.state_dict())