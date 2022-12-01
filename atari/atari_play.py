import gym
from gym.utils.play import play
    

def main():
    # set the atari game to play here
    game = "BattleZone-v5"

    # instantiates and enables user to play the game
    env = gym.make('ALE/'+game, render_mode="rgb_array", obs_type="ram")
    play(env,zoom=3)

if __name__ == '__main__':
    main()
