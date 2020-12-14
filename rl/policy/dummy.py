import numpy as np

class Dummy(object):
    def __init__(self):
        pass

    def set_eval(self):
        pass

    def set_train(self):
        pass

    def select_action(self, state):
        return np.ones( (len(state),), dtype=np.float32)

    def train(self, *args, **kwargs):
        return (0., 0., 0., 0.)

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

