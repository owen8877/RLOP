from unittest import TestCase

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as Func
from attr import dataclass
from matplotlib import pyplot as plt

from .bs import BSBaseline
from .nn_v6_cpu import GaussianPolicy_v6, PriceEstimator_v6_grad_fn


@dataclass
class QLBSFitResult:
    sigma: float
    mu: float
    estimated_prices: np.ndarray
    implied_vols: np.ndarray


class RLOPModel:
    def __init__(self, is_call_option: bool, checkpoint: str, anchor_T: float):
        assert is_call_option, "Only call option is supported in this model."
        self.nn_policy = GaussianPolicy_v6(is_call_option=is_call_option, from_filename=checkpoint)
        self.nn_price_estimator = PriceEstimator_v6_grad_fn(self.nn_policy.theta_mu, self.nn_policy.optimizer)
        self.anchor_T = anchor_T

    def sigma_transform(self, sigma_raw: torch.Tensor) -> torch.Tensor:
        # maps R -> (0.01, 5.01), same as original
        return torch.sigmoid(sigma_raw) * 5.0 + 0.01
        # return Func.elu(sigma_raw) + 1.01

    def _sigma_raw_from_guess(self, sigma_guess: float) -> float:
        """
        Invert sigma_transform approximately so that the *effective* σ
        starts near sigma_guess (instead of at a random point).
        """
        # sigma = 5*sigmoid(raw) + 0.01  => sigmoid(raw) = (sigma - 0.01)/5
        v = (sigma_guess - 0.01) / 5.0
        v = float(np.clip(v, 1e-4, 1 - 1e-4))
        return float(sp.special.logit(v))

    # ------------------------- FIT -------------------------

    def fit(
        self,
        spot: float,
        time_to_expiries: np.ndarray,
        strikes: np.ndarray,
        r: float,
        risk_lambda: float,
        friction: float,
        observed_prices: np.ndarray,
        weights: np.ndarray,
        sigma_guess: float,
        mu_guess: float,
        n_epochs: int = 200,
    ) -> QLBSFitResult:
        """
        Fit two scalars (sigma_raw, mu) so that the RLOP price estimator
        matches observed_prices under weighted MSE.

        NOTE: risk_lambda argument is currently ignored (kept 0 to match
        how this network was trained).
        """
        N = len(strikes)
        assert len(observed_prices) == N, "Length of observed_prices must match strikes"
        assert len(weights) == N, "Length of weights must match strikes"
        assert sigma_guess > 0

        # We optimize over sigma_raw; sigma_guess is an actual vol level
        sigma0_raw = self._sigma_raw_from_guess(sigma_guess)
        sigma = torch.tensor(sigma0_raw, requires_grad=True, dtype=torch.float32)
        mu = torch.tensor(mu_guess, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([sigma, mu], lr=0.05)

        # Scaling (same convention as QLBS for consistency)
        _scaled_strikes = torch.tensor(strikes / spot, dtype=torch.float32)
        _scaled_prices = torch.tensor(observed_prices / spot, dtype=torch.float32)
        _scaled_tte = torch.tensor(time_to_expiries / self.anchor_T, dtype=torch.float32)
        _scaled_rs = r * _scaled_tte
        _scaled_risk_lambdas = torch.full((N,), 0.0, dtype=torch.float32)  # RLOP was trained with λ≈0
        _scaled_frictions = torch.full((N,), friction, dtype=torch.float32)
        _scaled_weights = torch.tensor(weights * np.sqrt(spot), dtype=torch.float32)

        _zeros, _ones = torch.zeros(N, dtype=torch.float32), torch.ones(N, dtype=torch.float32)

        def get_state_info_tensor(mu_scalar: torch.Tensor, sigma_scalar: torch.Tensor) -> torch.Tensor:
            """
            Encode (T, μ, σ) exactly as in training:

              T_input  = anchor_T
              μ_input  = μ * (τ / anchor_T)
              σ_input  = σ( raw ) * sqrt(τ / anchor_T)

            where τ is actual time_to_expiry.
            """
            return torch.stack(
                [
                    _ones,                                 # spot (normalized to 1)
                    _zeros,                                # passed_real_time
                    _ones * self.anchor_T,                 # remaining_real_time (anchor horizon)
                    _scaled_strikes,                       # strike/spot
                    _scaled_rs,                            # r * (τ / anchor_T)
                    (mu_scalar * _scaled_tte),             # μ scaled by τ/anchor_T
                    (self.sigma_transform(sigma_scalar) * torch.sqrt(_scaled_tte)),  # σ scaled by sqrt(τ/anchor_T)
                    _scaled_risk_lambdas,                  # risk_lambda (≈0 here)
                    _scaled_frictions,                     # friction
                ],
                dim=1,
            )

        loss_history: list[float] = []

        for epoch in range(n_epochs + 1):
            optimizer.zero_grad()

            state_info_tensor = get_state_info_tensor(mu, sigma)
            estimated_price = self.nn_price_estimator.batch_estimate(state_info_tensor)  # shape [N]
            loss = torch.mean(_scaled_weights * (estimated_price - _scaled_prices) ** 2)  # type: ignore

            loss.backward()
            optimizer.step()

            l_val = loss.item()
            loss_history.append(l_val)

            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch}: loss={l_val:.6f}, "
                    f"sigma_eff={self.sigma_transform(sigma).item():.6f}, mu={mu.item():.6f}"
                )

            if l_val < 1e-7:
                print("Early stopping as loss is sufficiently small.")
                break

            # (You can optionally add a plateau early-stop here with a safe length check.)

        with torch.no_grad():
            state_info_tensor = get_state_info_tensor(mu, sigma)
            estimated_price, latent_vol = self.nn_price_estimator.batch_estimate(
                state_info_tensor, return_latent_vol=True  # type: ignore
            )

        est_price_np = (estimated_price.detach().cpu().numpy() * spot)
        _scaled_tte_np = _scaled_tte.numpy()
        latent_vol_np = latent_vol.detach().cpu().numpy()

        return QLBSFitResult(
            sigma=sigma.detach().cpu().item(),          # raw parameter (used again in predict)
            mu=mu.detach().cpu().item(),
            estimated_prices=est_price_np,
            implied_vols=latent_vol_np / np.sqrt(_scaled_tte_np),
        )

    # ------------------------- PREDICT -------------------------

    def predict(
        self,
        spot: float,
        time_to_expiries: np.ndarray,
        strikes: np.ndarray,
        r: float,
        risk_lambda: float,
        friction: float,
        sigma_fit: float,
        mu_fit: float,
    ) -> QLBSFitResult:
        """
        Price with previously fitted (sigma_fit, mu_fit).

        sigma_fit is the *raw* σ parameter returned by fit; we apply
        sigma_transform() inside here exactly as in fit, and we use the same
        T/μ/σ/r/friction scaling.
        """
        sigma = torch.tensor(sigma_fit, requires_grad=False, dtype=torch.float32)
        mu = torch.tensor(mu_fit, requires_grad=False, dtype=torch.float32)

        N = len(strikes)
        _scaled_strikes = torch.tensor(strikes / spot, dtype=torch.float32)
        _scaled_tte = torch.tensor(time_to_expiries / self.anchor_T, dtype=torch.float32)
        _scaled_rs = r * _scaled_tte
        _scaled_risk_lambdas = torch.full((N,), 0.0, dtype=torch.float32)
        _scaled_frictions = torch.full((N,), friction, dtype=torch.float32)

        _zeros, _ones = torch.zeros(N, dtype=torch.float32), torch.ones(N, dtype=torch.float32)

        def get_state_info_tensor(mu_scalar: torch.Tensor, sigma_scalar: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [
                    _ones,                                 # spot
                    _zeros,                                # passed_real_time
                    _ones * self.anchor_T,                 # remaining_real_time
                    _scaled_strikes,
                    _scaled_rs,
                    (mu_scalar * _scaled_tte),             # μ scaled by τ/anchor_T
                    (self.sigma_transform(sigma_scalar) * torch.sqrt(_scaled_tte)),  # σ scaled by sqrt(τ/anchor_T)
                    _scaled_risk_lambdas,
                    _scaled_frictions,
                ],
                dim=1,
            )

        with torch.no_grad():
            state_info_tensor = get_state_info_tensor(mu, sigma)
            estimated_price, latent_vol = self.nn_price_estimator.batch_estimate(
                state_info_tensor, return_latent_vol=True  # type: ignore
            )

        est_price_np = (estimated_price.detach().cpu().numpy() * spot)
        _scaled_tte_np = _scaled_tte.numpy()
        latent_vol_np = latent_vol.detach().cpu().numpy()

        return QLBSFitResult(
            sigma=sigma.detach().cpu().item(),
            mu=mu.detach().cpu().item(),
            estimated_prices=est_price_np,
            implied_vols=latent_vol_np / np.sqrt(_scaled_tte_np),
        )

    # ------------------------- Convenience -------------------------

    def fit_predict(
        self,
        spot: float,
        time_to_expiries: np.ndarray,
        strikes: np.ndarray,
        r: float,
        risk_lambda: float,
        friction: float,
        observed_prices: np.ndarray,
        weights: np.ndarray,
        sigma_guess: float,
        mu_guess: float,
        n_epochs: int = 200,
    ) -> QLBSFitResult:
        result = self.fit(
            spot=spot,
            time_to_expiries=time_to_expiries,
            strikes=strikes,
            r=r,
            risk_lambda=risk_lambda,
            friction=friction,
            observed_prices=observed_prices,
            weights=weights,
            sigma_guess=sigma_guess,
            mu_guess=mu_guess,
            n_epochs=n_epochs,
        )
        return self.predict(
            spot=spot,
            time_to_expiries=time_to_expiries,
            strikes=strikes,
            r=r,
            risk_lambda=risk_lambda,
            friction=friction,
            sigma_fit=result.sigma,
            mu_fit=result.mu,
        )


# ============================================================
# Tests (unchanged interface, will exercise the fixes)
# ============================================================

class TestTrainedModel(TestCase):
    def test_load_models(self):
        # setup hyperparameters
        r = 0.02         # interest rate, annualized
        friction = 2e-3  # transaction cost per unit traded
        is_call_option = True
        risk_lambda = 0.0
        T = 28 / 252
        spot = 10.0

        sigma_guess = 1.0
        mu_guess = -2.0

        def vol_curve_true(K):
            return 0.52 - 1.5 * (sp.special.expit(spot / K - 1) - 0.5)

        # synthesize test data
        N = 100
        Ks = np.linspace(0.85 * spot, 1.15 * spot, N)

        sigma_BS = vol_curve_true(Ks)
        bs_baseline = BSBaseline(is_call=is_call_option)
        target_price = bs_baseline.batch_estimate(
            torch.tensor(
                np.column_stack(
                    (
                        np.full(N, spot),        # spot
                        np.zeros(N),             # passed_real_time
                        np.full(N, T),           # remaining_real_time
                        Ks,                      # strike
                        np.full(N, r),           # r
                        np.zeros(N),             # mu
                        sigma_BS,                # sigma
                        np.full(N, risk_lambda), # risk_lambda
                        np.full(N, friction),    # friction
                    )
                )
            ).float()
        )  # type: ignore
        observed_prices = target_price.numpy() + np.random.randn(N) * 0.1

        # load trained models
        model = RLOPModel(is_call_option=is_call_option, checkpoint="trained_model/testr9/policy_1.pt", anchor_T=T)
        result = model.fit_predict(
            spot=spot,
            time_to_expiries=Ks * 0 + T,
            strikes=Ks,
            r=r,
            risk_lambda=risk_lambda,
            friction=friction,
            observed_prices=observed_prices,
            weights=np.ones(N),
            sigma_guess=sigma_guess,
            mu_guess=mu_guess,
            n_epochs=500,
        )

        # plot results
        fig, axs = plt.subplots(2, 1, figsize=(7, 5))

        axs[0].plot(Ks, observed_prices, label="Target Price with Noise", color="black", linestyle="--")
        axs[0].plot(Ks, result.estimated_prices, label="Estimated Price", color="blue", linestyle="-")
        axs[0].set_ylabel("Option Price")
        axs[0].legend(loc="best")

        axs[1].plot(Ks, vol_curve_true(Ks), label="True Vol Curve", color="green", linestyle="--")
        axs[1].plot(Ks, result.implied_vols, label="Inverted Latent Vol", color="red", linestyle="-")
        axs[1].set_ylabel("Volatility")
        axs[1].legend(loc="best")

        plt.xlabel("Strike Price")
        plt.tight_layout()
        plt.show()
