import numpy as np
import torch
import gym
import pygame
from pygame.locals import *
import sys
import os
from nn_init import Autoencoder

CHECKPOINT_PATH = "atari/models/epoch=66-step=7035.ckpt"

class AtariEnv:
    def __init__(self, game="BattleZone-v5", render="human", obs="ram"):
        self.env = gym.make('ALE/'+game, render_mode=render, obs_type=obs)  # atari environment
        self.obs = self.env.observation_space                               # observations
        self.action = self.env.action_space                                 # actions
        
        self.num_actions = self.action.n        # number of actions
        if os.path.isfile(CHECKPOINT_PATH) and os.path.splitext(CHECKPOINT_PATH)[-1].lower() == '.ckpt':
            print("Found pretrained model, loading...")
            model = Autoencoder.load_from_checkpoint(CHECKPOINT_PATH)
            self.encoder = model.encoder
            self.num_obs = 120
        else:
            print("No pretrained model found")
            self.encoder = None
            self.num_obs = 1        # number of observations
            for d in self.obs.shape:
                self.num_obs *= d
    
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
    
    def get_state_features(self, prev_state, curr_state):
        if not self.encoder:
            return None
        state = torch.stack((prev_state, curr_state),dim=1)
        encoded_state = self.encoder(state).squeeze()
        return encoded_state



    
# when this file is called, an atari game is simulated 
def main():
    # instantiate an atari game environment
    atari = AtariEnv("BattleZone-v5", obs="rgb")
    atari.env.reset(seed=0)

    # take actions in the game where each iteration in the loop represents one action
    episode_reward = 0  # total reward for the episode
    for i in range(1000):
        # take a random action
        action = atari.env.action_space.sample()
        action = 1
        observation, reward, terminated, truncated, info = atari.env.step(action)

        # print information about the game
        if i%50 == 0:
            print("-------------------- Iteration " + str(i) + " --------------------")
            print("Observation:\n", observation)
            print("Shape of Observations:\n",observation.shape)
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
