import gym

env = gym.make('Skiing-v0', render_mode='human')
env.reset()
for _ in range(100):
    env.step(env.action_space.sample())
env.close()