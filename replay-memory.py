import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import random
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from operator import add
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
plt.rcParams["figure.figsize"] = (20,15)

class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.pointer = 0
        self.experience = []
    
    def store_experience(self, experience):
        self.experience[self.pointer] = experience
        if self.pointer + 1 >= self.size:
            self.pointer = 0
        else:
            self.pointer += 1
        return
        
    def sample(self, exp_batch):
        indexes = random.sample(range(self.size), exp_batch)
        res = [val for i, val in enumerate(self.experience) if i in indexes]
        current_states = []
        actions = []
        rewards = []
        next_states = []
        for exp in res:
            current_states.append(torch.tensor(exp[0]))
            actions.append(torch.tensor(exp[1]))
            rewards.append(torch.tensor(exp[2]))
            next_states.append(torch.tensor(exp[3]))
        
        return current_states, actions, rewards, next_states
