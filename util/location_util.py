import numpy as np


# Objects can be distinquished by RGB codes.
# Player: [214, 92, 92]
# Flags (blue): [66, 72, 200]
# Flags (red): [184, 50, 50]

def get_pos_player(obs):
    ids = np.where(np.sum(obs == [214, 92, 92], -1) == 3)
    return ids[0].mean(), ids[1].mean()


def get_pos_flags(obs):
    obs_clip = obs[:obs.shape[0] // 2]
    if np.any(np.sum(obs == [184, 50, 50], -1) == 3):
        ids = np.where(np.sum(obs == [184, 50, 50], -1) == 3)
        return ids[0].mean(), ids[1].mean()
    else:
        base = 0
        ids = np.where(np.sum(obs_clip == [66, 72, 200], -1) == 3)
        return ids[0].mean() + base, ids[1].mean()


def get_speed(observe, observe_old):
    min_val = np.inf
    min_idx = 0
    for k in range(0, 7):
        val = np.sum(np.abs(observe[54:-52, 8:152] - observe_old[54 + k:-52 + k, 8:152]))
        if min_val > val:
            min_idx = k
            min_val = val
    return min_idx
