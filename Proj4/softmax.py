import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # This uses the last axis, which is where classes are expected to be ((N, K) shape).
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)