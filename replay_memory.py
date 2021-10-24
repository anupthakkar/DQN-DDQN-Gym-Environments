import numpy as np
import random
from operator import add
import torch


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.pointer = 0
        self.experience = []
    
    # this function helps with storing the experiences
    def store_experience(self, experience):
        
        # if the experience buffer is less than the size, add it directly to the experience buffer, else append it wrt to the pointer in a round-robin way.
        if len(self.experience) < self.size:
            self.experience.append(experience)
        
        else:
            self.experience[self.pointer] = experience
        
        # update the pointer based on the size.
        self.pointer = (self.pointer+1) % self.size
    
    # this function returns the sample from the experience buffer.
    def sample(self, exp_batch):
        current_states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        # if the size of the experience buffer is less than the batch size, return the experience buffer as is, else randomly sample as per the batch size.
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
