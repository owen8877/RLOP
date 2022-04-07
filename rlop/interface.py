import numpy as np


class Policy:
    def __init__(self):
        pass

    def action(self, state, info):
        raise NotImplementedError

    def update(self, delta: np.sctypes, action, state, info, *args):
        raise NotImplementedError


class Baseline:
    def __init__(self):
        pass

    def __call__(self, state, info):
        raise NotImplementedError

    def update(self, delta: np.sctypes, state, info):
        raise NotImplementedError
