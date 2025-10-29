import time
from unittest import TestCase

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.util import ensure_dir
from lib.util.net import CombinedResNet
from lib.util.pricing import bs_euro_vanilla_call, delta_hedge_bs_euro_vanilla_call


def in_transform(x: torch.Tensor) -> torch.Tensor:
    # x[:, 0].log_()  # spot
    # x[:, 3].log_()  # strike
    return x


def out_transform(y: torch.Tensor) -> torch.Tensor:
    y[:, 0] = -torch.exp(y[:, 0])  # price
    y[:, 1] = torch.sigmoid(y[:, 1])  # delta
    return y


class Test(TestCase):
    def test_get_pretrained_nn(self):
        # configs
        num_epochs = 100
        batch_size = 100_000

        train_n, test_n = 1_000_000, 100_000

        net = CombinedResNet(
            input_dim=9,
            hidden_dim=64,
            transform_pair=(in_transform, out_transform),
            activation="elu",
            groups=3,
            layer_per_group=3,
        ).cuda()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

        # heavylifting
        if not torch.cuda.is_available():
            print("Warning: No GPU found, using CPU instead.")

        def get_Xy(n: int):
            K = np.exp(np.random.rand(n) * 2 - 1)
            t_pass = np.random.rand(n) * 1.9 + 0.1
            risk_lambda = np.random.rand(n) * 1.0
            friction = np.random.rand(n) * 0.01
            mu = np.random.randn(n) * 1.0
            T = np.random.rand(n) * 1.9 + 0.1
            sigma = np.random.rand(n) * 0.9 + 0.1
            r = np.random.rand(n) * 0.05

            dplus = np.random.randn(n) * 1.5
            S = K * np.exp(dplus * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T)

            def tr(S):
                return (-(mu - sigma**2 / 2) * (T - t_pass) + np.log(S)) / sigma

            X = np.stack(
                [
                    tr(S),  # normal_price
                    t_pass,  # passed_real_time
                    T,  # remaining_real_time
                    tr(K),  # normal_strike_price
                    r,  # r
                    mu,  # mu
                    sigma,  # sigma
                    risk_lambda,  # risk_lambda
                    friction,  # friction
                ],
                axis=1,
            )

            y = bs_euro_vanilla_call(S, K, T, r, sigma)
            delta = delta_hedge_bs_euro_vanilla_call(S, K, T, r, sigma)
            Y = np.stack([y, delta], axis=1)

            return torch.tensor(X, dtype=torch.float32).cuda(), torch.tensor(Y, dtype=torch.float32).cuda()

        X_train, y_train = get_Xy(train_n)
        X_test, y_test = get_Xy(test_n)

        def criterion(y_pred, y_true):
            price_pred, delta_pred = y_pred[:, 0], y_pred[:, 1]
            price_true, delta_true = y_true[:, 0], y_true[:, 1]

            price_loss = torch.mean((torch.log(-price_pred + 1e-8) - torch.log(price_true + 1e-8)) ** 2)
            # price_loss = torch.mean((price_pred + price_true) ** 2)
            delta_loss = torch.mean((torch.logit(delta_pred, 1e-3) - torch.logit(delta_true, 1e-3)) ** 2)

            return price_loss + delta_loss

        num_batches = X_train.size(0) // batch_size

        writer = SummaryWriter("runs/pretrain_qlbs")

        for epoch in range(num_epochs):
            net.train()
            epoch_loss = 0.0
            start_time = time.time()
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                inputs = X_train[start_idx:end_idx]
                targets = y_train[start_idx:end_idx]

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= num_batches

            net.eval()
            with torch.no_grad():
                test_outputs = net(X_test)
                test_loss = criterion(test_outputs, y_test).item()

            elapsed_time = time.time() - start_time
            writer.add_scalar("Loss/Train", epoch_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)

            # add memory usage
            writer.add_scalar("Memory/Allocated (MB)", torch.cuda.memory_allocated() / (1024 * 1024), epoch)
            writer.add_scalar("Memory/Cached (MB)", torch.cuda.memory_reserved() / (1024 * 1024), epoch)

            if np.isnan(epoch_loss) or np.isinf(epoch_loss):
                print("Loss is NaN or Inf, stopping training.")
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss:.6f}, Time: {elapsed_time:.2f}s"
                )

        writer.close()
        ensure_dir("trained_model")
        torch.save(net.state_dict(), "trained_model/pretrained-qlbs.pt")

    def test_sanity_check(self):
        net = CombinedResNet(
            input_dim=9,
            hidden_dim=64,
            transform_pair=(in_transform, out_transform),
            activation="elu",
            groups=3,
            layer_per_group=3,
        )
        net.load_state_dict(torch.load("trained_model/pretrained-qlbs.pt"))
        net.eval()

        # visualize the predicted price by setting strike=10, r=0.01, sigma=0.2, T=1, and varying spot from 1 to 20
        import matplotlib.pyplot as plt

        S = np.linspace(0.2, 5, 100)
        K = 0 * S + 1
        t_pass = 0 * S + 0
        risk_lambda = 0 * S + 1.0
        friction = 0 * S + 0
        mu = 0 * S + 0.05
        T = 0 * S + 0.8
        sigma = 0 * S + 0.3
        r = 0 * S + 0.02

        def tr(S):
            return (-(mu - sigma**2 / 2) * (T - t_pass) + np.log(S)) / sigma

        X = np.stack(
            [
                tr(S),  # normal_price
                t_pass,  # passed_real_time
                T,  # remaining_real_time
                tr(K),  # normal_strike_price
                r,  # r
                mu,  # mu
                sigma,  # sigma
                risk_lambda,  # risk_lambda
                friction,  # friction
            ],
            axis=1,
        )
        print(X)

        price_bs = bs_euro_vanilla_call(S, K, T, r, sigma)
        delta_bs = delta_hedge_bs_euro_vanilla_call(S, K, T, r, sigma)
        output_net = net(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        price_net, delta_net = output_net[:, 0], output_net[:, 1]

        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        axs[0].plot(S, price_bs, label="BS")
        axs[0].plot(S, -price_net, label="NN")
        axs[0].set(ylabel="Call Option Price", title="Call Option Price vs Spot Price")
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(S, delta_bs, label="BS")
        axs[1].plot(S, delta_net, label="NN")
        axs[1].set(xlabel="Spot Price", ylabel="Call Option Delta", title="Call Option Delta vs Spot Price")
        axs[1].legend()
        axs[1].grid()

        plt.show()
