import numpy as np


class Policy:
    def __init__(self):
        pass

    def action(self, state, info):
        raise NotImplementedError

    def batch_action(self, state_info_tensor):
        """
        :param state_info_tensor: [[normal_price, strike_price, r, mu, sigma, remaining_real_time, risk_lambda]]
        :return:
        """
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


class InitialEstimator:
    def __init__(self, is_call_option: bool):
        self.is_call_option = is_call_option

    def __call__(self, initial_asset_price: float, strike_price: float, remaining_time: int, r: float,
                 sigma: float, _dt: float) -> float:
        pass
