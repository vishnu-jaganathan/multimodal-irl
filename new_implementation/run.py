"""
Implementation of TAMER (Knox + Stone, 2009)
Inspired by: https://github.com/benibienz/TAMER
"""

import gym

from tamer.agent import Tamer
from tamer.transcribe import get_nlp_score


if __name__ == '__main__':
    env = gym.make('ALE/Bowling-v5', render_mode="human")

    # hyperparameters
    num_episodes = 2
    feedback_interval = 5
    tamer_training_timestep = 0.5

    agent = Tamer(env, num_episodes, tamer_training_timestep, feedback_interval)
    
    # why is it training for 2 episodes, when do games end, removed await, what does it do
    agent.train(model_file_to_save='autosave')
    agent.play(n_episodes=1, render=True)
    agent.evaluate(n_episodes=30)
