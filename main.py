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
from environment import GridEnvironment
from network import Net
from replay_memory import ReplayMemory

class Main:
    def __init__(self):
        self.training_episodes = 1000
        self.epsilon_decay = 0.001
        self.epsilon = 1
        self.discount_factor = 0.9
        self.env = GridEnvironment('deterministic')
        self.neural_net = Net()
        self.learning_target = Net()
        self.memory = ReplayMemory(1000)
        self.batch_size = 4

    def select_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.env.action_space.n)
        else:
            actions = self.neural_net.predict(observation)
            action = actions.index(max(actions))
        self.epsilon = self.epsilon - (self.epsilon * self.epsilon_decay)
        return action
    
    def main(self):
        for epi in range(self.training_episodes):
            current_state = self.env.reset()
            action = self.select_action(current_state)
            observation, current_reward, done, info = self.env.step(action)
            self.memory.store_experience([current_state, action, current_reward, observation, done])
            current_state = observation
            while not done:
                action = self.select_action(current_state)
                observation, current_reward, done, info = self.env.step(action)
                self.memory.store_experience([current_state, action, current_reward, observation, done])
                current_state = observation
                current_states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                self.train(current_states, actions, rewards, next_states, dones)



