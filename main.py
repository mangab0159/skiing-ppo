import gym

from Agent import *
from util.location_util import *
from util.util import *

env_name = "Skiing-v4"
env = gym.make(env_name, render_mode="human")

n_cont = 4
obss = np.empty((0, 64, 64, 3 * n_cont))
acts = np.empty((0))

lr = 0.0002
agent = Agent(CNN(n_cont=n_cont), lr=lr)

for episode in range(1000):
    observe = env.reset()
    step = 0
    cnt = 0
    done = False
    r_a, c_a = get_pos_player(observe)
    r_f, c_f = get_pos_flags(observe)
    r_a_old, c_a_old = r_a, c_a
    observe_old = observe

    history = np.concatenate([pre_processing(observe)] * n_cont, -1)

    outs_o = []
    outs_a = []
    while not done:
        step += 1

        # TEACHER
        v_f = np.arctan2(r_f - r_a, c_f - c_a)  # direction from player to target
        spd = get_speed(observe, observe_old)
        v_a = np.arctan2(spd, c_a - c_a_old)  # speed vector of the player

        r_a_old, c_a_old = r_a, c_a
        observe_old = observe

        if spd == 0 and (c_a - c_a_old) == 0:
            # no movement
            cnt += 1
            act_t = np.random.choice(3, 1)[0]
        else:
            cnt = 0
            if v_f - v_a < -0.1:
                act_t = 1
            elif v_f - v_a > 0.1:
                act_t = 2
            else:
                act_t = 0

        if cnt > 10:
            print('no movement!')
            break

        outs_o.append(history)
        outs_a.append(act_t)

        act = agent.get_action(np.array([history]))

        observe, reward, done, info = env.step(act)

        history = np.concatenate([pre_processing(observe), history[:, :, :-3]], -1)
        r_a, c_a = get_pos_player(observe)
        r_f, c_f = get_pos_flags(observe)

    # append data & limit data size
    obss = np.concatenate([obss, outs_o], 0)
    acts = np.concatenate([acts, outs_a], 0)
    if len(obss) > 5000:
        obss = obss[-5000:]
        acts = acts[-5000:]

    if torch.cuda.is_available():
        agent.model.cuda()

    for i in range(500):
        d_x, d_y = batch(obss, acts)
        loss = agent.update(d_x, d_y)
        print('%5d %5d' % (episode, i), loss, end='\r')
    print()

    if torch.cuda.is_available():
        agent.model.cpu()

observe = env.reset()
done = False
history = np.concatenate([pre_processing(observe)] * n_cont, -1)

while not done:
    act = agent.get_action(np.array([history]))
    observe, reward, done, info = env.step(act)
    history = np.concatenate([pre_processing(observe), history[:, :, :-3]], -1)
