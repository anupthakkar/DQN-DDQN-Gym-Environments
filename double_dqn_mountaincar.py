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
plt.rcParams["figure.figsize"] = (10,10)
# from environment import GridEnvironment
from network_mountaincar import Net
from replay_memory import ReplayMemory
from tqdm import tqdm
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Double_dqn_mountaincar:
    def __init__(self, env_variable, env_type, input_size, output_size):
        # Here we are setting the parameters for the training and the neural networks.
        self.training_episodes = 2000
        self.epsilon_decay = 0.004
        self.epsilon = 1
        self.discount_factor = 0.99
        self.env = env_variable
        
        self.neural_net = Net(input_size, output_size)
        self.target_network = Net(input_size, output_size)
        self.target_network.transfer_weights(self.neural_net)
        self.memory = ReplayMemory(1000)
        self.batch_size = 64
        self.update_time_steps = 3
        self.env_type = env_type
        self.input_size = input_size
        self.output_size = output_size

    def select_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.env.action_space.n)
            # action = torch.tensor(action).to(device)
            # print(action)
        else:
            observation = torch.from_numpy(observation).float().to(device)

            actions = self.neural_net.forward(observation)
            actions.to(device)
            action = torch.argmax(actions)

        return action

    
    def main(self):
        replay_memory_size = [10000]

        for size in replay_memory_size:
            self.neural_net = Net(self.input_size, self.output_size)
            self.target_network = Net(self.input_size, self.output_size)
            self.target_network.transfer_weights(self.neural_net)
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
                done = False
                # while the episode hasnt terminated
                timestep_count = 0
                count +=1
                while not done:
                    
                    timestep_count += 1
                    # get the selected action from the current_state
                    action = self.select_action(current_state)
                    
                    # return the observations from that action - new state, the observed reward, etc
                    # print('action', action)
                    # action = action.numpy()
                    # action = action.cpu().detach().numpy()
                    if torch.is_tensor(action):
                        action = action.cpu().numpy()
                    observation, current_reward, done, info = self.env.step(action)
                    total_reward += current_reward

                    # store the experience in the replay buffer.
                    self.memory.store_experience([current_state, action, current_reward, observation, done])
                    current_state = observation

                    # sample the replay buffer and get the necessary objects.
                    current_states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                    
                    # if no. of samples is > batch size, we train the neural network.
                    if(timestep_count % 5 == 0):
                      if (len(current_states) >= self.batch_size):# and (epi != self.training_episodes-1):
                          self.train(current_states, actions, rewards, next_states, dones)
                    
                    # transfer the weights if we cross the specfied threshold.
                if count%self.update_time_steps == 0:
                    self.target_network.transfer_weights(self.neural_net)
                
                # decay the epsilon
                self.epsilon = self.epsilon - (self.epsilon * self.epsilon_decay)

                # store the necessary objectives.
                total_rewards_array.append(total_reward)
                all_time_steps.append(timestep_count)
                epsilons.append(self.epsilon)
                if(np.max(all_time_steps[-20:]) < 110):
                    break
                # print("Rewards for episode: {}, Timesteps: {}".format(total_reward, self.env.timestep))
            plt.figure()
            plt.plot(total_rewards_array)
            plt.title('{}: Total Rewards during Training in memory size {}'.format(self.env_type, size))
            plt.show()
            plt.figure()
            plt.plot(all_time_steps)
            plt.title('{}: Timesteps during Training in memory size {}'.format(self.env_type, size))
            plt.show()
            plt.figure()
            plt.plot(epsilons)
            plt.title('{}: Epsilon Decay during Training in memory size {}'.format(self.env_type, size))
            plt.show()

            self.test()

    def train(self, current_states, actions, rewards, next_states,dones):

        for index, s in enumerate(current_states):
          # print(type(s))
          current_states[index] = torch.tensor(s).float()
        for index, s in enumerate(next_states):
          next_states[index] = torch.tensor(s).float()
        for index, s in enumerate(actions):
          actions[index] = torch.tensor(s)
        current_states = torch.stack(current_states, dim=0)
        next_states = torch.stack(next_states, dim=0)
        actions = torch.stack(actions,dim=0)

        # for done, reward,target_next_state_prediction,nn_next_state_prediction in zip(dones,rewards,target_next_state_predictions,nn_next_state_predictions):
        #     # if done:
        #     #     targetpred = reward.float()
        #     #     all_rewards.append(targetpred)
        #     # else:
        #     # print('target_next_state_prediction',target_next_state_prediction)
        #     action_max = torch.argmax(target_next_state_prediction).to(device)
        #     # print('action_max',action_max)
        #     q_value_pred = nn_next_state_prediction[action_max]
        #     targetpred = reward + self.discount_factor * q_value_pred
        #     all_rewards.append(targetpred)

        target_max_actions = torch.argmax((self.neural_net(next_states)), dim = 1)
        # target_max_actions = self.target_network(next_states).argmax(dim=1).values

        # print('target_max_actions shape: ',target_max_actions.shape)
        # print('target_max_actions type: ',type(target_max_actions))

        q_target = self.target_network.forward(next_states)

        y_target = list()
        for index, value in enumerate(q_target):
          y_value = value[target_max_actions[index]]
          if dones[index]:
            y_target.append(torch.as_tensor(rewards[index]))
          else:
            y_target.append(torch.as_tensor(rewards[index] + self.discount_factor * y_value))
        
        q_pred = self.neural_net.forward(current_states)

        # print('y_pred shape: ',y_pred.shape)
        # print('y_pred type: ',type(y_pred))
        y_pred = list()
        for index, value in enumerate(q_pred):
            y_pred.append(value[actions[index]])

        y_target = torch.stack(y_target,dim=0)
        y_pred = torch.stack(y_pred,dim=0)
                
        criterion = nn.MSELoss()
        loss = criterion(y_target,y_pred)
        # losses.append(loss)
        self.neural_net.optimizer.zero_grad()
        loss.backward()
        self.neural_net.optimizer.step()
    

    # in this function, we are simply taking the greedy action and testing the policy learnt by the Neural network.
    def test(self):
        total_rewards_array = []
        all_time_steps = []
        self.epsilon = 0
        observation = self.env.reset()
        
        for i in tqdm(range(100)):
            observation = self.env.reset()
            total_reward = 0
            done = False
            timestep_count = 0
            while not done:
                timestep_count += 1
                action = self.select_action(observation)
                if torch.is_tensor(action):
                        action = action.cpu().numpy()
                observation, current_reward, done, info = self.env.step(action)
                total_reward+=current_reward
            total_rewards_array.append(total_reward)
            all_time_steps.append(timestep_count)
            # print("for {} iteration, the cumulative reward is {}".format(i,total_reward))
        
        plt.figure()
        plt.plot(total_rewards_array)
        plt.title('{}: Total Rewards during Testing'.format(self.env_type))
        plt.show()
        plt.figure()
        plt.plot(all_time_steps)
        plt.title('{}: Timesteps during Testing'.format(self.env_type))
        plt.show()

# mountaincar_obj = gym.make("MountainCar-v0")
# main_obj_mountaincar = Double_dqn_mountaincar(mountaincar_obj, 'Mountain Car DDQN', 2, 3)
# main_obj_mountaincar.main()