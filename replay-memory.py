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
        
    def sample(self, batch):
        indexes = random.sample(self.experience, batch)