import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter

import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

from .interface import Interface


BOWLING_ACTION_MAP = { 0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE" }

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')


class SGDFunctionApproximator:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample().reshape(-1) for _ in range(10000)], dtype='float64'
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        # applies these in parallel and concatenates results
        # note: this is to give nonlinearity to classifiers
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        # makes a model for every action in the action space
        # doing a partial fit on the base features
        # the partial fit does one sgd epoch
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(env.reset().reshape(-1))], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        # predicts reward for given state, action
        # if action isnt specified it returns this for all actions
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        # one sgd epoch on given action for the state features to the given reward
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        # applies the scaling and rbf kernels on the functions above
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """

    def __init__(
        self,
        env,
        num_episodes,
        ts_len=0.5,  # length of timestep for training TAMER
        feedback_interval=5, # give feedback per this many steps
        output_dir=LOGS_DIR,
        model_file_to_load=None  # filename of pretrained model
    ):

        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            self.H = SGDFunctionApproximator(env)  # init H function

        # hyperparameters
        self.num_episodes = num_episodes
        self.ts_len = ts_len
        self.feedback_interval = feedback_interval

        # reward logging
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        """ Get all actions and find max """
        return np.argmax(self.H.predict(state))

    def _train_episode(self, episode_index, disp):
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        rng = np.random.default_rng()
        tot_reward = 0
        state = self.env.reset().reshape(-1)
        ep_start_time = dt.datetime.now().time()
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                # print(f' {ts}', end='')

                # Determine next action
                action = self.act(state)

                # show action on the display panel
                disp.show_action(action)

                # Get next state and reward
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.reshape(-1)

                if ts % self.feedback_interval == 0: # every 5 timesteps allow for feedback.
                    now = time.time()
                    # length of time paused for feedback
                    while time.time() < now + self.ts_len:
                        frame = None

                        time.sleep(0.01)  # save the CPU

                        human_reward = disp.get_scalar_feedback()
                        feedback_ts = dt.datetime.now().time()
                        if human_reward != 0:
                            # if feedback is given int he window, it maps the model of (state,action) and human feedback
                            dict_writer.writerow(
                                {
                                    'Episode': episode_index + 1,
                                    'Ep start ts': ep_start_time,
                                    'Feedback ts': feedback_ts,
                                    'Human Reward': human_reward,
                                    'Environment Reward': reward
                                }
                            )
                            self.H.update(state, action, human_reward)
                            break

                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    break

                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state


    def train(self, model_file_to_save=None):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """

        disp = Interface(action_map=BOWLING_ACTION_MAP)

        for i in range(self.num_episodes):
            # one run of the game is an episdoe
            self._train_episode(i, disp)

        print('\nCleaning up...')
        self.env.close()
        if model_file_to_save is not None:
            self.save_model(filename=model_file_to_save)

    def play(self, n_episodes=1, render=False):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        ep_rewards = []
        for i in range(n_episodes):
            state = self.env.reset().reshape(-1)
            done = False
            tot_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.reshape(-1)
                tot_reward += reward
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        self.env.close()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return avg_reward

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H model from models dir
        Args:
            filename: name of pickled file
        """
        # loads the trained sklearn model
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            self.H = pickle.load(f)