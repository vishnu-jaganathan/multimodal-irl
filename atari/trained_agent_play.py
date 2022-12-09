import pickle
import json
import torch

from atari_env import AtariEnv


def main():
    model = pickle.load(open('atari/models/trained_agent.pkl', 'rb'))
    game = "Bowling-v5"
    atari = AtariEnv(game=game)

    with open('atari/ram_annotations.json') as f:
        ram_annotations = json.load(f)
    feature_indices = []
    for k in ram_annotations[game.lower().split("-")[0]]:
        if isinstance(ram_annotations[game.lower().split("-")[0]][k], list):
            feature_indices += ram_annotations[game.lower().split("-")[0]][k]
        else:
            feature_indices.append(ram_annotations[game.lower().split("-")[0]][k])
    input_dim = len(feature_indices) + 1

    atari.env.reset()
    features = torch.zeros(input_dim)
    done = False
    step_index = 0
    while not done:
        step_index += 1
        action = atari.choose_action(features, model)
        _, _, terminated, truncated, _ = atari.env.step(action)
        done = terminated or truncated

if __name__ == '__main__':
    main()
