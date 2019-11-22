import gym

env = gym.make('BipedalWalker-v2')
print(env.action_space)
print(env.observation_space)