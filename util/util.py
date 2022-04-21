import numpy as np
from skimage.transform import resize


def pre_processing(observe):
    processed_observe = resize(observe[54:-52, 8:152], (64, 64), mode='reflect', anti_aliasing=True)
    return processed_observe


def batch(obss, acts, batch_size=32):
    n_data = len(obss)
    ids = np.random.choice(n_data, batch_size, replace=False)
    b_o = obss[ids]
    b_a = acts[ids]
    return b_o, b_a
