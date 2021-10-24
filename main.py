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
from tqdm import tqdm
from torch.autograd import Variable

class Main:
    def __init__(self):
        # Here we are setting the parameters for the training and the neural networks.
        self.training_episodes = 1000
        self.epsilon_decay = 0.01
        self.epsilon = 1
        self.discount_factor = 0.9
        self.env = GridEnvironment('deterministic')
        self.neural_net = Net()
        self.learning_target = Net()
        self.learning_target.transfer_weights(self.neural_net)
        self.memory = ReplayMemory(10000)
        self.batch_size = 40
        self.update_time_steps = 5

    # The epsilon greedy method for selecting the actions.
    def select_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.env.action_space.n)

        else:
            observation = torch.from_numpy(observation).float()
            actions = self.neural_net.forward(observation)
            actions = actions.detach().numpy().tolist()
            action = actions.index(max(actions))

        return action
    
    def main(self):
        replay_memory_size = [40, 100, 250, 500, 1000, 10000]

        for size in replay_memory_size:
            self.memory = ReplayMemory(size)
            self.epsilon = 1
            count = 0
            total_rewards_array = []
            all_time_steps = []
            epsilons = []
            for epi in tqdm(range(self.training_episodes)):
                # print("EPISODE {}".format(epi))
                total_reward = 0
                current_state = self.env.reset()

                # while the episode hasnt terminated
                while not self.env.done:
                    count +=1
                    # get the selected action from the current_state
                    action = self.select_action(current_state)
                    
                    # return the observations from that action - new state, the observed reward, etc
                    observation, current_reward, done, info = self.env.step(action)
                    total_reward += current_reward

                    # store the experience in the replay buffer.
                    self.memory.store_experience([current_state, action, current_reward, observation, done])
                    current_state = observation

                    # sample the replay buffer and get the necessary objects.
                    current_states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                    
                    # if no. of samples is > batch size, we train the neural network.
                    if (len(current_states) >= self.batch_size):# and (epi != self.training_episodes-1):
                        self.train(current_states, actions, rewards, next_states, dones)
                    
                    # transfer the weights if we cross the specfied threshold.
                    if count%self.update_time_steps ==0:
                        self.learning_target.transfer_weights(self.neural_net)
                
                # decay the epsilon
                self.epsilon = self.epsilon - (self.epsilon * self.epsilon_decay)

                # store the necessary objectives.
                total_rewards_array.append(total_reward)
                all_time_steps.append(self.env.timestep)
                epsilons.append(self.epsilon)
                # print("Rewards for episode: {}, Timesteps: {}".format(total_reward, self.env.timestep))
            plt.figure()
            plt.plot(total_rewards_array)
            plt.title('Total Rewards during Training in memory size {}'.format(size))
            plt.show()
            plt.figure()
            plt.plot(all_time_steps)
            plt.title('Timesteps during Training in memory size {}'.format(size))
            plt.show()
            plt.figure()
            plt.plot(epsilons)
            plt.title('Epsilon Decay during Training in memory size {}'.format(size))
            plt.show()

            self .test()

    # here we are actually calcuating the y_pred and doing the backprop based on the loss.
    def train(self, current_states, actions, rewards, next_states,dones):

        all_rewards = []
        losses = []

        for current_state_main,done,reward,next_state_main,action in zip(current_states,dones,rewards,next_states,actions):
            
            # if we have reached the reward state the ypred is the reward as is, or we calculate the discounted reward. 
            if done:
                targetpred  = reward.float()
                all_rewards.append(targetpred)
            
            else:
                next_state = next_state_main.float()
                target_predictions = self.learning_target.forward(next_state)
                targetpred = reward + self.discount_factor * torch.max(target_predictions)
                all_rewards.append(targetpred)

            # doing backprop and training the policy network.
            current_state = current_state_main.float()
            policy_predictions = self.neural_net.forward(current_state)
            criterion = nn.MSELoss()
            loss = criterion(targetpred,policy_predictions[action])
            losses.append(loss)
            self.neural_net.optimizer.zero_grad()
            loss.backward()

            # this is just a basic optimization which helps the neural network converge faster (similar to what relu and other activation functions are doing.)
            # for example clamp(min=0) = ReLU
            for param in self.neural_net.parameters():
                param.grad.data.clamp_(-1, 1)
            
            self.neural_net.optimizer.step() 

    # in this function, we are simply taking the greedy action and testing the policy learnt by the Neural network.
    def test(self):
        total_rewards_array = []
        all_time_steps = []
        self.epsilon = 0
        observation = self.env.reset()
        
        for i in range(10):
            observation = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                
                action = self.select_action(observation)
                observation, current_reward, done, info = self.env.step(action)
                total_reward+=current_reward
            total_rewards_array.append(total_reward)
            all_time_steps.append(self.env.timestep)
            # print("for {} iteration, the cumulative reward is {}".format(i,total_reward))
        
        plt.figure()
        plt.plot(total_rewards_array)
        plt.title('Total Rewards during Testing')
        plt.show()
        plt.figure()
        plt.plot(all_time_steps)
        plt.title('Timesteps during Testing')
        plt.show()

main_obj = Main()
main_obj.main()