import numpy as np
import torch
import gym
import pygame
from pygame.locals import *
import sys

class AtariEnv:
    def __init__(self, game="BattleZone-v5", render="rgb_array",obs="ram"):
        self.env = gym.make('ALE/'+game, render_mode=render, obs_type=obs)
        self.obs = self.env.observation_space
        self.action = self.env.action_space

        self.num_features = self.obs.shape[0]
        self.num_actions = self.action.n
    
    def choose_action(self, features, model):
        action_rewards = model(features)
        return np.argmax(action_rewards)
        
    

def main():
    atari = AtariEnv("BattleZone-v5")
    atari.env.reset(seed=0)
    episode_reward = 0
    for i in range(1000):
        action = atari.env.action_space.sample()
        observation, reward, terminated, truncated, info = atari.env.step(action)
        print(torch.Tensor(list(observation) + [3]))
        print(torch.Tensor(list(observation) + [3]).shape)
        print(torch.Tensor(list(observation) + [3])[3])
        

        break
        if i%50 == 0:
            print("-------------------- Iteration " + str(i) + " --------------------")
            print("Observation:\n", observation)
            print("Information:", info, "\n")
        episode_reward += reward
        if terminated or truncated:
            break
    print('Reward: %s' % episode_reward)

if __name__ == '__main__':
    main()
