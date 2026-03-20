import torch

from .env import Policy, Baseline
from lib.util.pricing import (
    bs_euro_vanilla_call,
    bs_euro_vanilla_put,
    delta_hedge_bs_euro_vanilla_put,
    delta_hedge_bs_euro_vanilla_call,
)


class BSPolicy(Policy):
    def __init__(self, is_call: bool = True):
        super().__init__()
        self.is_call = is_call

    def action(self, state, info, return_pre_action=False):
        return self.batch_action(state.to_tensor(info), return_pre_action=return_pre_action)

        # sits = []
        # for t in range(self.passed_step, self.passed_step + self.remaining_step):
        #     N = len(self.spots)
        #     sit = torch.empty((N, 9))
        #     sit[:, 0] = torch.tensor(self.spots)  # spot
        #     sit[:, 1] = t * info._dt  # passed_real
        #     sit[:, 2] = (self.remaining_step + self.passed_step - t) * info._dt  # remaining_real_time
        #     sit[:, 3] = torch.tensor(info.strike_prices)  # strike
        #     sit[:, 4] = info.r  # r
        #     sit[:, 5] = torch.tensor(info.mus)  # mu
        #     sit[:, 6] = torch.tensor(info.sigmas)  # sigma
        #     sit[:, 7] = torch.tensor(info.risk_lambdas)  # risk_lambda
        #     sit[:, 8] = torch.tensor(info.frictions)  # friction
        #     sits.append(sit)
        # return torch.concat(sits, dim=0)

        # S = state.spots
        # K = info.strike_prices
        # _T = 0 * S + state.remaining_step
        # _d = (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
        #     S, K, _T, info.r, info.sigmas, info._dt
        # )
        # if return_pre_action:
        #     return torch.tensor(_d), torch.tensor(info.sigmas)
        # else:
        #     return torch.tensor(_d)

    def batch_action(
        self, state_info_tensor: torch.Tensor, random: bool = True, return_pre_action=False
    ) -> torch.Tensor:
        """
        :param state_info_tensor: [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda]]
        :return:
        """
        passed_real_time = state_info_tensor[:, 1]
        remaining_real_time = state_info_tensor[:, 2]
        r = state_info_tensor[:, 4]
        mu = state_info_tensor[:, 5]
        sigma = state_info_tensor[:, 6]
        S = state_info_tensor[:, 0]
        K = state_info_tensor[:, 3]

        _d = (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
            S, K, remaining_real_time, r, sigma
        )
        if isinstance(_d, torch.Tensor):
            if return_pre_action:
                return _d.detach().clone(), sigma.detach().clone()  # type: ignore
            else:
                return _d.detach().clone()  # type: ignore

        else:
            if return_pre_action:
                return torch.tensor(_d), sigma.detach().clone()  # type: ignore
            else:
                return torch.tensor(_d)  # type: ignore

    def update(self, delta: float, action, state, info, *args):
        raise Exception("BS policy cannot be updated!")


class BSBaseline(Baseline):
    def __init__(self, is_call: bool):
        super().__init__()
        self.is_call = is_call

    def __call__(self, state, info):
        S = state.spots
        K = info.strike_prices
        _T = 0 * S + state.remaining_step
        _d = (bs_euro_vanilla_call if self.is_call else bs_euro_vanilla_put)(S, K, _T, info.r, info.sigmas, info._dt)
        return torch.tensor(_d)

    def batch_estimate(self, state_info_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param state_info_tensor: [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda]]
        :return:
        """
        passed_real_time = state_info_tensor[:, 1]
        remaining_real_time = state_info_tensor[:, 2]
        r = state_info_tensor[:, 4]
        mu = state_info_tensor[:, 5]
        sigma = state_info_tensor[:, 6]
        S = state_info_tensor[:, 0]
        K = state_info_tensor[:, 3]

        _d = (bs_euro_vanilla_call if self.is_call else bs_euro_vanilla_put)(S, K, remaining_real_time, r, sigma, 1)

        return _d  # type: ignore

    def update(self, G: float, state, info):
        raise Exception("BS baseline cannot be updated!")
