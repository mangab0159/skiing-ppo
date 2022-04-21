import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


def pre_processing(observe):
    processed_observe = resize(observe[54:-52, 8:152], (64, 64), mode='reflect', anti_aliasing=True)
    return processed_observe


def batch(obss, acts, batch_size=32):
    n_data = len(obss)
    ids = np.random.choice(n_data, batch_size, replace=False)
    b_o = obss[ids]
    b_a = acts[ids]
    return b_o, b_a


def save_plot(returns_list, loss_list):
    plt.subplots(constrained_layout=True)
    plt.subplot(2, 1, 1)
    plt.plot(loss_list)

    plt.title('mean loss per episode')
    plt.xlabel('episodes')
    plt.ylabel('loss_mean')

    plt.subplot(2, 1, 2)
    plt.plot(returns_list)

    plt.title('returns per episode')
    plt.xlabel('episodes')
    plt.ylabel('returns')

    plt.savefig('./plots/loss_returns_plot.png')
