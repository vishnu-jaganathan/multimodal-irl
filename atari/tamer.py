import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import json
import sys
from time import time

from atari_env import AtariEnv
from nn_init import LinearNN

from multiprocessing import Process, Queue
import keyboard


# use cuda if available
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Using {device} device")


# print debug
DEBUG = True

# reward provided to the agent
KEY_LABEL = 0.0

# mappings from action integer to string describing the action that was taken
ACTIONS = { 0: "NOOP",
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


'''
Keeps track of the keys that are pressed and puts the reward and time of keypress into key_queue
Inputs:
    key_queue   queue to store key presses
'''
def key_tracker(key_queue):
    while True:
        key = keyboard.read_key()   # key that was pressed
        t = time()                  # time of keypress
        if key in {"1","2","3","4","5","6","7","8","9","0","down"}:
            keydown(key)                    # translate key that was pressed to a reward
            key_queue.put((KEY_LABEL,t))    # add y into the queue (y from Deep TAMER paper where y = (reward,time))

'''
Translate the key press to a reward
Inputs:
    key     string representing the key that was pressed
'''
def keydown(key):
    global KEY_LABEL

    # set KEY_LABEL based on the key that was pressed
    if key == "0":
        KEY_LABEL = 4
    elif key == "9":
        KEY_LABEL = 3
    elif key == "8":
        KEY_LABEL = 2
    elif key == "7":
        KEY_LABEL = 1
    elif key == "6":
        KEY_LABEL = 0
    elif key == "5":
        KEY_LABEL = -1
    elif key == "4":
        KEY_LABEL = -2
    elif key == "3":
        KEY_LABEL = -3
    elif key == "2":
        KEY_LABEL = -4
    elif key == "1":
        KEY_LABEL = -5
    elif key == "down":
        KEY_LABEL = "TERMINATE"

    # do nothing until key is released
    while keyboard.is_pressed(key):
        pass



def main():
    global KEY_LABEL
    global ACTIONS
    global DEBUG

    game = "Bowling-v5"

    with open('atari/ram_annotations.json') as f:
            ram_annotations = json.load(f)
    feature_indices = []
    for k in ram_annotations[game.lower().split("-")[0]]:
        if isinstance(ram_annotations[game.lower().split("-")[0]][k], list):
            feature_indices += ram_annotations[game.lower().split("-")[0]][k]
        else:
            feature_indices.append(ram_annotations[game.lower().split("-")[0]][k])

    # instantiate the atari game environment
    atari = AtariEnv(game=game)
    if game == "Bowling-v5":
        atari.num_actions = 4
        ACTIONS[3] = "DOWN"

    # NN Hyperparameters
    hidden_dims = [16, 16, 16]          # hidden layers where each value in the list represents the number of nodes in each layer
    input_dim = len(feature_indices)    # input is the features where features=[observations from OpenAI RAM, action]
    output_dim = atari.num_actions      # output is the predicted reward for each action
    alpha = 1e-5                        # learning rate

    # NN
    model = LinearNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=alpha)
    model.zero_grad()

    episodes = 5            # number of episodes (i.e. full games) to play
    kill = False            # boolean indicating whether or not to stop all training
    for e in range(episodes):
        if kill: break                      # stop all episodes
        atari.env.reset()                   # reset environment
        features = torch.zeros(input_dim)   # initialize features
        trajectory = []                     # initialize trajectory containing experience {x} (state x from Deep TAMER paper where x = [observations, action, start time, end time])

        done = False            # boolean indicating whether or not the game is done
        step_index = 0
        interval = 1
        while not done:
            step_index += 1
            action = atari.choose_action(features, model)
            observation, _, terminated, truncated, _ = atari.env.step(action)
            if DEBUG:
                print("action:", int(action), ACTIONS[int(action)])

            done = terminated or truncated  # whether or not the episode has finished

            # state x from the Deep TAMER paper where x = [observations, action]
            x = torch.Tensor(np.array(observation)[feature_indices].tolist())
            print(x)

            # take an action based on reward model
            if step_index % interval == 0:
                while True:
                    key = keyboard.read_key()
                    if key in {"1","2","3","4","5","6","7","8","9","0","down"}:
                        break
                keydown(key)
            
                # stop if commanded to terminate
                if KEY_LABEL == "TERMINATE":
                    kill = True
                    break

                # train NN
                n = len(trajectory)
                if KEY_LABEL != 0 and n > interval:
                    for i in range(n-interval,n):
                        x_i = trajectory[i][0]
                        reward = model(x_i)                    # forward propagation
                        loss = (KEY_LABEL - reward[action]).pow(2)  # loss
                        model.zero_grad()                           # zero gradients
                        loss.backward()                             # compute gradients
                        optimizer.step()                            # SGD
                        # print debug information
                        if DEBUG:
                            print("reward:", reward.tolist())
                            print("reward[action]:", float(reward[int(action)]))
                            print("feedback:", float(KEY_LABEL))
                            print("loss:", float(loss))

            trajectory.append((x,KEY_LABEL))            # add state x to trajectory where x = [observations, action]
            features = x               # set features for next state as [observations, action]

    pickle.dump(model, open('atari/models/trained_agent_vanilla.pkl', 'wb'))    # save model
        

if __name__ == "__main__":
    main()