import numpy as np
import random
from operator import add
import torch


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.pointer = 0
        self.experience = []
    
    def store_experience(self, experience):
        
        if len(self.experience) < self.size:
            self.experience.append(experience)
        
        else:
            self.experience[self.pointer] = experience
        
        self.pointer = (self.pointer+1) % self.size
        
    def sample(self, exp_batch):
        current_states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        if(len(self.experience) <= exp_batch):

            return current_states, actions, rewards, next_states, dones
        else:

            res = random.sample(self.experience, exp_batch)
        
        for exp in res:
            current_states.append(torch.tensor(exp[0]))
            actions.append(torch.tensor(exp[1]))
            rewards.append(torch.tensor(exp[2]))
            next_states.append(torch.tensor(exp[3]))
            dones.append(torch.tensor(exp[4]))

        return current_states, actions, rewards, next_states, dones
