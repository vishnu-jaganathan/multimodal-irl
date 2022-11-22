import gym
from gym.utils.play import play
    

def main():
    game = "BattleZone-v5"
    env = gym.make('ALE/'+game, render_mode="rgb_array", obs_type="ram")
    
    play(env,zoom=3)

if __name__ == '__main__':
    main()
