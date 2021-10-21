import numpy as np
import matplotlib.pyplot as plt
from operator import add
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size=25, output_size=4,lr=1e-3):
        super(Net, self).__init__()

        # the architecture of the model is 
        # 25 * 64 * 128 * 256 * 4
        self.linear_batchnorm_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64, bias=True,),
            # nn.BatchNorm1d((64,1)),
            nn.ReLU(),
            nn.Linear(64, 128, bias=True),
            # nn.BatchNorm1d((128,1)),
            nn.ReLU(),
            nn.Linear(128, 256, bias=True),
            # nn.BatchNorm1d((128,1)),
            nn.ReLU(),
            nn.Linear(256, output_size, bias=True),
            nn.ReLU()
        )

    def predict(self, x):
        # x = torch.flatten(x)
        output = self.linear_batchnorm_relu_stack(x)
        return nn.Softmax(output)
    
    def save_model(self,filename='models/temporary_model.pth'):
        torch.save(self.state_dict(), filename)

    def load_model(self,filename='models/temporary_model.pth'):
        self.load_state_dict(torch.load(filename))

# Testing the file functions
# input1 = np.zeros(25)
# input1[2] = 1
# final_input = torch.from_numpy(input1).float()
# model = Net()
# print(final_input)
# print(model.predict(final_input))
# model.save_model()
# print(model)
# print('model1')
# model2 = Net()
# model2.load_model()
# print(model2)
# print('model2')