import gym

from Agent import *
from util.util import *

lr = 0.0002
n_count = 4
agent = Agent(CNN(n_count=4), lr=lr)

PATH = './weights/'
agent.model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))
agent.model.eval()

env_name = "Skiing-v4"
env = gym.make(env_name, render_mode="human")

observe = env.reset()
done = False
history = np.concatenate([pre_processing(observe)] * n_count, -1)

while not done:
    act = agent.get_action(np.array([history]))
    observe, reward, done, info = env.step(act)
    history = np.concatenate([pre_processing(observe), history[:, :, :-3]], -1)
