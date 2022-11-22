import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import sys
from time import time,sleep

from atari_env import AtariEnv
from nn_init import LinearNN
from loss import WeightedMSE

from multiprocessing import Process, Queue
import keyboard


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Using {device} device")


KEY_LABEL = 0.0

def keydown(key):
    global KEY_LABEL

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

    while keyboard.is_pressed(key):
        pass


def key_tracker(key_queue):
    while True:
        key = keyboard.read_key()
        t = time()
        if key in {"1","2","3","4","5","6","7","8","9","0","down"}:
            keydown(key)
            key_queue.put((KEY_LABEL,t))

def main():
    global KEY_LABEL

    key_queue = Queue()
    key_process = Process(target=key_tracker, args=(key_queue,))
    key_process.start()

    atari = AtariEnv(game="BattleZone-v5")

    # NN Hyperparameters
    hidden_dims = [16,16]
    input_dim = atari.num_features  # number of input features where features are the observations
    output_dim = atari.num_actions  # reward for each action
    alpha = 1e-2

    # NN
    # input: features
    # output: reward for each action
    model = LinearNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)
    # criterion = WeightedMSE()
    optimizer = optim.SGD(model.parameters(), lr=alpha)
    model.zero_grad()


    episodes = 5

    # D = {(x,y,w)}
    feedback_buffer = []
    for e in range(episodes):
        
        atari.env.reset()
        done = False
        features = np.zeros(atari.num_features)
        trajectory = []
        
        t_start = time()
        t_state_start = t_start
        while not done:
            action = atari.choose_action(features, model)
            observation, _, terminated, truncated, _ = atari.env.step(action)
            t_state_end = time()

            # x = [features, action, start time, end time]
            x = torch.Tensor(list(observation) + [action] + [t_state_start-t_start, t_state_end-t_start])
            done = terminated or truncated

            while key_queue.qsize():
                y, t_feed = key_queue.get()
                t_feed -= t_start
                if y == "TERMINATE":
                    done = True
                    key_process.close()
                    key_process.join()
                    break
                
                # assign feedback y to all states x in the interval [t_f-4, t_f-0.2]
                i = len(trajectory)-1
                while i >= 0:
                    if trajectory[i][2] <= t_feed-4 <= trajectory[i][3] or trajectory[i][2] <= t_feed-0.2 <= trajectory[i][3]:
                        if trajectory[i][2] <= t_feed-4 <= trajectory[i][3] and trajectory[i][2] <= t_feed-0.2 <= trajectory[i][3]:
                            weight = 1
                        elif trajectory[i][2] <= t_feed-4 <= trajectory[i][3]:
                            weight = (trajectory[i][3] - (t_feed-4)) / 3.8
                        elif trajectory[i][2] <= t_feed-0.2 <= trajectory[i][3]:
                            weight = ((t_feed-0.2) - trajectory[i][3]) / 3.8
                        
                        reward = model(trajectory[i])
                        print(type(reward))


                        target = torch.clone(reward)
                        target[action] = y
                        loss = weight*(target - reward).pow(2)
                        print(loss)
                        loss.backward()

                        feedback_buffer.append((trajectory[i], y, weight))

                    i -= 1

                trajectory.append(x)


        atari.env.step(action)
    pickle.dump(model, open('trained_agent.pkl', 'wb'))

        
        

if __name__ == "__main__":
    main()