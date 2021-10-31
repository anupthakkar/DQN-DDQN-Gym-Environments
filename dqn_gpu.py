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
from network import Net
from replay_memory import ReplayMemory
from tqdm import tqdm
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Main:
    def __init__(self, env_variable, env_type, input_size, output_size):
        # Here we are setting the parameters for the training and the neural networks.
        self.training_episodes = 10000
        self.epsilon_decay = 0.001
        self.epsilon = 1
        self.discount_factor = 0.95
        self.env = env_variable
        self.neural_net = Net(input_size, output_size)
        self.learning_target = Net(input_size, output_size)
        self.learning_target.transfer_weights(self.neural_net)
        self.memory = ReplayMemory(1000000)
        self.batch_size = 128
        self.update_time_steps = 10
        self.env_type = env_type
        self.input_size = input_size
        self.output_size = output_size

    # The epsilon greedy method for selecting the actions.
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
                timestep_count = 0
                count +=1
                # while the episode hasnt terminated
                while not done:
                    timestep_count += 1
                    # get the selected action from the current_state
                    action = self.select_action(current_state)
                    # print(action)
                    # return the observations from that action - new state, the observed reward, etc
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
                    if (len(current_states) >= self.batch_size):# and (epi != self.training_episodes-1):
                        self.train(current_states, actions, rewards, next_states, dones)
                    
                    # transfer the weights if we cross the specfied threshold.
                if count%self.update_time_steps ==0:
                    self.learning_target.transfer_weights(self.neural_net)
                
                # decay the epsilon
                self.epsilon = self.epsilon - (self.epsilon * self.epsilon_decay)

                # store the necessary objectives.
                total_rewards_array.append(total_reward)
                all_time_steps.append(timestep_count)
                epsilons.append(self.epsilon)
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

    # here we are actually calcuating the y_pred and doing the backprop based on the loss.
    def train(self, current_states, actions, rewards, next_states,dones):

        # all_rewards = []
        # losses = []
        # # print('current_states 1 - ',len(current_states),current_states,next_states)
        # current_states = torch.stack(current_states, dim=0).to(device).float()
        # next_state_main = torch.stack(next_states, dim=0).to(device).float()
        # print('rewards: ',rewards)
        # print('Dones: ',dones)
        # rewards = torch.stack(rewards,dim=0).to(device).float()
        # # print('current_states',current_states.shape,current_states,next_states)
        # target_next_state_predictions = self.target_network.forward(next_state_main)
        # nn_current_state_predictions = self.neural_net.forward(current_states)
        # # print("target_next_state_predictions",target_next_state_predictions.shape)
        # # print('rewards',rewards)
        # # print('target_next_state_predictions',target_next_state_predictions.shape)
        # # print('dones',dones)
        # dones = torch.stack(dones,dim=0).to(device).float()
        # # print('dones',dones)
        # # print('self.discount_factor' ,self.discount_factor )
        # # all_rewards = rewards + self.discount_factor * torch.argmax(target_next_state_predictions, axis=1) * dones
        # # print(all_rewards)
        # for done, reward,target_next_state_prediction in zip(dones,rewards,target_next_state_predictions):
        #     if done:
        #         targetpred = reward.float()
        #         all_rewards.append(targetpred)
        #     else:
        #         action_max = torch.argmax(target_next_state_prediction).to(device)
        #         q_value_pred = torch.max(target_next_state_prediction)
        #         targetpred = reward + self.discount_factor * q_value_pred
        #         all_rewards.append(targetpred)
        # print(all_rewards)
        # criterion = nn.MSELoss()
        # new_rewards = []
        # for reward in all_rewards:
        #     new_rewards.append(reward.to(device))
        # new_actions = []
        # for action in actions:
        #     new_actions.append(action.to(device))
        
        # targetpred = torch.stack(new_rewards,dim=0)
        # targetpred = targetpred.to(device)
        # new_actions = torch.stack(new_actions,dim=0)
        # new_actions = new_actions.to(device)
        # temp_temp = torch.index_select(nn_current_state_predictions,1,new_actions)
        # # doing backprop and training the policy network.
        # loss = criterion(temp_temp,targetpred)
        # losses.append(loss)
        # self.neural_net.optimizer.zero_grad()
        # loss.backward()
        # # print(next(self.neural_net.parameters()).grad.data.clamp_(-1,1))
        # # for param in self.neural_net.parameters():
        #   # param.grad.data.clamp_(0, 100)
        # self.neural_net.optimizer.step()
        for index, s in enumerate(current_states):
          current_states[index] = torch.tensor(s.float())
        for index, s in enumerate(next_states):
          next_states[index] = torch.tensor(s.float())
        for index, s in enumerate(actions):
          actions[index] = torch.tensor(s)
        current_states = torch.stack(current_states,dim=0)
        next_states = torch.stack(next_states, dim=0)
        actions = torch.stack(actions, dim=0)
        q_pred = self.neural_net.forward(current_states).gather(1, actions.view(-1,1))
        target_q_values = self.target_network.forward(next_states).max(dim=1).values

        y_target = list()
        for index, value in enumerate(target_q_values):
            if dones[index]:
             y_target.append(rewards[index])
            else:
                y_target.append(rewards[index] + self.discount_factor * value)
        
        y_target = torch.stack(y_target,dim=0)
        # print(y_target)
        criterion = nn.MSELoss()
        loss = criterion(y_target,q_pred)
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
        
        for i in range(10):
            observation = self.env.reset()
            total_reward = 0
            done = False
            timestep_count = 0
            while not done:
                timestep_count += 1
                action = self.select_action(observation)
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

# main_obj = Main()
# main_obj.main()

cartpole_obj = gym.make("CartPole-v1")
main_obj_cartpole = Main(cartpole_obj, 'Cartpole Environment', 4, 2)
main_obj_cartpole.main()