import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import json
import sys
from time import time,sleep

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
    global KEY_LABEL

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

    # with open('atari/ram_annotations.json') as f:
    #         ram_annotations = json.load(f)
    # feature_indices = []
    # for k in ram_annotations[game.lower().split("-")[0]]:
    #     if isinstance(ram_annotations[game.lower().split("-")[0]][k], list):
    #         feature_indices += ram_annotations[game.lower().split("-")[0]][k]
    #     else:
    #         feature_indices.append(ram_annotations[game.lower().split("-")[0]][k])

    key_queue = Queue() # queue containing y based on key presses (y from Deep TAMER paper where y = (reward,time))

    # start process on another thread that tracks key presses 
    key_process = Process(target=key_tracker, args=(key_queue,))
    key_process.start()

    # instantiate the atari game environment
    atari = AtariEnv(game=game, obs="grayscale")
    if game == "Bowling-v5":
        atari.num_actions = 4
        ACTIONS[3] = "DOWN"


    # NN Hyperparameters
    hidden_dims = [16,16]               # hidden layers where each value in the list represents the number of nodes in each layer
    input_dim = atari.num_obs + 1  # input is the features where features = [grayscale, action]
    output_dim = atari.num_actions      # output is the predicted reward for each action
    alpha = 1e-7                        # learning rate

    # NN
    model = LinearNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)
    for module in model.weights:
        module.weight.data.fill_(0)
        module.bias.data.fill_(0)
        
    optimizer = optim.SGD(model.parameters(), lr=alpha)
    model.zero_grad()

    episodes = 5            # number of episodes (i.e. full games) to play
    feedback_buffer = []    # feedback replay buffer D from the Deep TAMER paper where D = {(x,y,w)} (currently not used)
    kill = False            # boolean indicating whether or not to stop all training
    for e in range(episodes):
        if kill: break                      # stop all episodes
        atari.env.reset()                   # reset environment
        prev_state = torch.zeros((1,210,160))  # initialize features
        curr_state = torch.zeros((1,210,160))  # initialize features
        state = atari.get_state_features(prev_state, curr_state)
        features = torch.cat((state,torch.ones(1)))
        trajectory = []                     # initialize trajectory containing experience {x} (state x from Deep TAMER paper where x = [observations, action, start time, end time])

        done = False            # boolean indicating whether or not the game is done
        t_start = time()        # start time of episode
        t_state_start = 0 # current state start time

        while not done:
            # take an action based on reward model
            action = atari.choose_action(features, model)
            curr_state, _, terminated, truncated, _ = atari.env.step(action)

            done = terminated or truncated  # whether or not the episode has finished
            
            t_state_end = time() - t_start    # current state end time (and next state start time)

            # state x from the Deep TAMER paper where x = [observations, action, start time, end time]
            curr_state = torch.Tensor(curr_state)[None,:,:]
            state = atari.get_state_features(prev_state, curr_state)
            x = torch.Tensor(state.tolist() + [action] + [t_state_start, t_state_end])
            # remove all feedback y in the queue (y from Deep TAMER paper where y = (reward,time))
            while key_queue.qsize():
                h, t_feed = key_queue.get() # h from Deep TAMER paper where h = true reward
                                            # t_feed = t^f from Deep TAMER paper where t^f = time of feedback

                # stop if commanded to terminate
                if h == "TERMINATE":
                    kill = True
                    done = True
                    key_process.terminate()     # kill key tracking process
                    key_process.join()
                    break

                # feedback time interval [t^f - 4, t^f - 0.2] where t^f = time of feedback
                t_feed -= t_start
                t_feed_start = t_feed - 2
                t_feed_end = t_feed - 0.2

                # assign feedback y = (reward,time) to all states x in the feedback time interval [t^f - 4, t^f - 0.2] where t^f = time of feedback
                i = len(trajectory)-1
                # print debug information
                if DEBUG:
                    print("Trajectory Length:", i)
                # start from most current state in trajectory and loop backwards
                while i >= 0:
                    x_i = trajectory[i]
                    
                    # time interval that the state occurred
                    t_i_state_start = x_i[-2]
                    t_i_state_end = x_i[-1]

                    # if the end of the ith state in the trajectory >= start of the feedback time interval, then it is possible we assign the feedback to the ith state in the trajectory
                    # otherwise, the ith state in the trajectory is not in the feedback time interval and all states before the ith state are also not in the feedback time interval, so break from the loop
                    if t_i_state_end >= t_feed_start:

                        # assign the feedback to the ith state in the trajectory if the start or end of the ith state in the trajectory is within the feedback time interval
                        if (t_feed_start <= t_i_state_start <= t_feed_end) or (t_feed_start <= t_i_state_end <= t_feed_end):
                            # calculate w from the Deep TAMER paper where w = weight applied to loss function
                            # assume f_delay has a uniform distribution over the feedback time interval [t^f - 4, t^f - 0.2] where t^f = time of feedback 
                            t_overlap = min(t_i_state_end,t_feed_end) - max(t_i_state_start,t_feed_start)
                            weight = t_overlap / (4-0.2)

                            # train NN
                            reward = model(x_i[:-2])                    # forward propagation
                            loss = weight*(h - reward[action]).pow(2)   # loss
                            model.zero_grad()                           # zero gradients
                            loss.backward()                             # compute gradients
                            optimizer.step()                            # SGD
                            # print debug information
                            if DEBUG:
                                print("action:", int(action), ACTIONS[int(action)])
                                print("reward:", reward.tolist())
                                print("reward[action]:", float(reward[int(action)]))
                                print("feedback:", float(h))
                                print("loss:", float(loss))

                            # add (x,y,w) to feedback replay buffer D
                            feedback_buffer.append((trajectory[i], h, weight))

                    else: break

                    i -= 1  # move backwards through trajectory

            trajectory.append(x)            # add state x to trajectory where x = [observations, action, start time, end time]
            t_state_start = t_state_end     # set start of next state as end of current state
            features = x[:-2]               # set features for next state as [observations]
            prev_state = torch.clone(curr_state)

    pickle.dump(model, open('atari/models/trained_agent.pkl', 'wb'))    # save model
    key_process.terminate()                                             # kill key tracking process
    key_process.join()
    
        

if __name__ == "__main__":
    main()