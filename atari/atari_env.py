import numpy as np
import torch
import gym
import pygame
from pygame.locals import *
import sys

class AtariEnv:
    def __init__(self, game="BattleZone-v5", render="human", obs="ram"):
        self.env = gym.make('ALE/'+game, render_mode=render, obs_type=obs)  # atari environment
        self.obs = self.env.observation_space                               # observations
        self.action = self.env.action_space                                 # actions

        self.num_obs = self.obs.shape[0]        # number of observations
        self.num_actions = self.action.n        # number of actions
    
    '''
    Chooses the action that maximizes the predicted reward as the next action
    Inputs:
        features    array containing features of the current state
        model       pytorch reward model
        debug       boolean on whether or not to print debug information
    Outputs:
        action      int representing the action to take
    '''
    def choose_action(self, features, model, debug=False):
        action_rewards = model(features)        # prediction of the rewards for each action
        action = torch.argmax(action_rewards)   # action returned is the index of the maximum reward

        # print debug information
        if debug:
            print("action_rewards:", action_rewards.tolist())
            print("action:", float(action))

        return int(action)



    
# when this file is called, an atari game is simulated 
def main():
    # instantiate an atari game environment
    atari = AtariEnv("BattleZone-v5")
    atari.env.reset(seed=0)

    # take actions in the game where each iteration in the loop represents one action
    episode_reward = 0  # total reward for the episode
    for i in range(1000):
        test_sequence = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
        for action in test_sequence:
            observation, reward, terminated, truncated, info = atari.env.step(action)
        return
        # take a random action
        action = atari.env.action_space.sample()
        action = 1
        observation, reward, terminated, truncated, info = atari.env.step(action)

        # print information about the game
        if i%50 == 0:
            print("-------------------- Iteration " + str(i) + " --------------------")
            print("Observation:\n", observation)
            print("Information:", info, "\n")
        
        # add the reward for taking the action
        episode_reward += reward
        
        # end game
        if terminated or truncated:
            break
    
    # print reward of the episode
    print('Reward: %s' % episode_reward)

if __name__ == '__main__':
    main()
