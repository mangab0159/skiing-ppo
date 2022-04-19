import gym
import torch

from Agent import Agent
from location_util import *

min_score = -40000
proxy_score = 500
# dist_pass = 20
dist_pass = 30
done = False

gamma = 0.99
lr = 0.0002
agent = Agent(lr)#.cuda()

# game setting
env_name = "Skiing-v4"
env = gym.make(env_name, render_mode="human")

n_eps = 100000
update_every = 1

xs_batch, acts_batch, rs_batch, returns_batch = [], [], [], []

for ep in range(1, n_eps + 1):
    obs = env.reset()
    rs = []
    xs = []
    acts = []
    done2 = False
    while not done:
        # do something to environment
        x = (torch.from_numpy(obs).unsqueeze(0).float().permute(0, 3, 1, 2) / 255) * 2 - 1
        act = agent.get_action(x)
        # act = agent.get_action(x.cuda()).cpu()

        obs, r1, done, info = env.step(act)

        # get r2
        x_p, y_p = get_pos_player(obs)
        x_f, y_f = get_pos_flags(obs)

        dist = (y_p - y_f) ** 2 #((x_p - x_f) ** 2 + (y_p - y_f) ** 2)
        # print(dist)
        if np.isnan(dist):
            dist = 0
        r2 = -dist
        if x_p > x_f:
            if ((x_p - x_f) ** 2 + (y_p - y_f) ** 2) ** 0.5 < dist_pass:
                #r2 = proxy_score
                done2 = True # 전체 학습땐 주석처리
            else:
                done2 = True
        if done:
            r = (r1 - min_score) + r2
        else:
            r = r2

        xs.append(x)
        acts.append(act)
        # print(r)
        rs.append(r)
        # print((x_p, y_p), (x_f, y_f), act, r)
        # print(x_p, x_f, r)
        if done2:
            break

    xs = torch.cat(xs, 0)
    acts = torch.LongTensor(acts)
    rs = [r / len(rs) for r in rs]
    # rs = torch.FloatTensor(rs)

    gs = []
    g = 0
    for r in rs[::-1]:
        g = r + gamma * g
        gs.insert(0, g)
    gs = torch.FloatTensor(gs)

    agent.cuda()
    agent.update_episodes(xs.cuda(), acts.cuda(), gs.cuda(), use_norm=True)
    agent.cpu()
    # agent.update_episodes(xs, acts, gs)

    # xs_batch.append(xs)
    # acts_batch.append(acts)
    # rs_batch.append(gs)

    # if ep % update_every == 0:
    #     xs_batch = torch.cat(xs_batch, 0)
    #     acts_batch = torch.cat(acts_batch, 0)
    #     rs_batch = torch.cat(rs_batch, 0)

    #     agent.update_episodes(xs_batch, acts_batch, rs_batch)

    #     xs_batch, acts_batch, rs_batch, returns_batch = [], [], [], []

    print('(ep, gs): ({0}, {1})'.format(ep, gs[0]))

# logits = agent(xs)
...


PATH = './weights/'

torch.save(agent.model, PATH + 'model.pt')  # 전체 모델 저장
torch.save(agent.model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
torch.save({
    'model': agent.model.state_dict(),
    'optimizer': agent.opt.state_dict()
}, PATH + 'all.tar')

env.close()
