import numpy as np


# Objects can be distinquished by RGB codes.
# Player: [214, 92, 92]
# Flags (blue): [66, 72, 200]
# Flags (red): [184, 50, 50]

def get_pos_player(obs):
    ids = np.where(np.sum(obs == [214, 92, 92], -1) == 3)
    return ids[0].mean(), ids[1].mean()


def get_pos_flags(obs):
    # obs_clip = obs[:obs.shape[0] // 2]
    if np.any(np.sum(obs == [184, 50, 50], -1) == 3):
        ids = np.where(np.sum(obs == [184, 50, 50], -1) == 3)
        return ids[0].mean(), ids[1].mean()
    else:
        base = 0
        ids = np.where(np.sum(obs == [66, 72, 200], -1) == 3)
        return ids[0].mean() + base, ids[1].mean()
