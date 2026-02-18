# src/models/lm_mlp.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class LMConfig:
    epochs_max: int = 100
    stop_error: float = 1e-5       # MSE threshold
    stop_grad: float = 1e-5        # ||J^T r|| threshold
    mu_init: float = 1e-3          # initial damping
    mu_dec: float = 0.1            # decrease factor when accept
    mu_inc: float = 10.0           # increase factor when reject
    mu_max: float = 1e12           # cap for numerical safety
    max_damping_tries: int = 10    # per-epoch inner tries


class LMMLP:
    """
    84–J–1 MLP with sigmoid hidden + sigmoid output, trained by LM on MSE loss.

    y_hat = sigmoid( sigmoid(XW1 + b1) W2 + b2 )
    Targets y are 0/1 (you will map -1->0 outside).
    """

    def __init__(self, J: int, device: torch.device, dtype: torch.dtype = torch.float64, seed: int = 0, D: int = 84):
        self.J = int(J)
        self.device = device
        self.dtype = dtype
        self.D = int(D)

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        # Xavier-ish init
        D = self.D
        W1 = torch.randn(self.J, D, generator=g, dtype=torch.float64) * np.sqrt(2.0 / (D + self.J))
        b1 = torch.zeros(self.J, dtype=torch.float64)
        w2 = torch.randn(self.J, generator=g, dtype=torch.float64) * np.sqrt(2.0 / (self.J + 1))
        b2 = torch.zeros(1, dtype=torch.float64)

        self.W1 = W1.to(device=device, dtype=dtype).contiguous()
        self.b1 = b1.to(device=device, dtype=dtype).contiguous()
        self.w2 = w2.to(device=device, dtype=dtype).contiguous()
        self.b2 = b2.to(device=device, dtype=dtype).contiguous()

    # ---------- basic ops ----------
    @staticmethod
    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N,D)
        return yhat: (N,)
        """
        a1 = X @ self.W1.t() + self.b1  # (N,J)
        h = self._sigmoid(a1)           # (N,J)
        a2 = h @ self.w2 + self.b2      # (N,)
        y = self._sigmoid(a2)           # (N,)
        return y

    def num_params(self) -> int:
        # W1: J*D, b1: J, w2: J, b2: 1
        return self.J * self.D + self.J + self.J + 1  # 86J + 1

    def get_param_vector(self) -> torch.Tensor:
        """
        Flatten params to (P,) with fixed order:
          [W1 (row-major), b1, w2, b2]
        """
        return torch.cat([
            self.W1.reshape(-1),
            self.b1.reshape(-1),
            self.w2.reshape(-1),
            self.b2.reshape(-1),
        ])

    def set_param_vector(self, v: torch.Tensor) -> None:
        """
        Inverse of get_param_vector
        """
        v = v.to(device=self.device, dtype=self.dtype).reshape(-1)
        J = self.J
        D = self.D
        nW1 = J * D
        nb1 = J
        nw2 = J
        nb2 = 1

        p0 = 0
        W1 = v[p0:p0 + nW1].reshape(J, D); p0 += nW1
        b1 = v[p0:p0 + nb1].reshape(J); p0 += nb1
        w2 = v[p0:p0 + nw2].reshape(J); p0 += nw2
        b2 = v[p0:p0 + nb2].reshape(1); p0 += nb2

        self.W1 = W1.contiguous()
        self.b1 = b1.contiguous()
        self.w2 = w2.contiguous()
        self.b2 = b2.contiguous()

    # ---------- LM core ----------
    def _jacobian(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic Jacobian of yhat wrt parameters.
        Returns:
          yhat: (N,)
          Jmat: (N,P)
        """
        # Forward intermediates
        a1 = X @ self.W1.t() + self.b1         # (N,J)
        h = torch.sigmoid(a1)                  # (N,J)
        d1 = h * (1.0 - h)                     # (N,J)

        a2 = h @ self.w2 + self.b2             # (N,)
        yhat = torch.sigmoid(a2)               # (N,)
        d2 = yhat * (1.0 - yhat)               # (N,)

        N = X.shape[0]
        J = self.J
        P = self.num_params()
        Jmat = torch.zeros((N, P), device=self.device, dtype=self.dtype)

        # block order: W1, b1, w2, b2
        # dy/dw2 = d2 * h
        # dy/db2 = d2
        # dy/dW1_{j,k} = d2 * w2_j * d1[:,j] * X[:,k]
        # dy/db1_j      = d2 * w2_j * d1[:,j]

        # --- W1 block ---
        D = self.D
        col = 0
        for j in range(J):
            scale = d2 * self.w2[j] * d1[:, j]  # (N,)
            Jmat[:, col:col + D] = scale[:, None] * X  # (N,D)
            col += D

        # --- b1 block ---
        for j in range(J):
            Jmat[:, col] = d2 * self.w2[j] * d1[:, j]
            col += 1

        # --- w2 block ---
        Jmat[:, col:col + J] = d2[:, None] * h
        col += J

        # --- b2 block ---
        Jmat[:, col] = d2
        col += 1

        return yhat, Jmat

    def fit_lm(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        cfg: LMConfig,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        model_select_metric: str = "val_acc",
        patience: int = 15,
        threshold: float = 0.5,
    ) -> dict:
        """
        Full-batch LM training on train set.
        If val provided: keep best model by val metric (acc or auc) with patience.
        Returns training summary dict.
        """
        # sanity
        X_train = X_train.to(self.device, self.dtype)
        y_train = y_train.to(self.device, self.dtype).reshape(-1)

        if X_val is not None:
            X_val = X_val.to(self.device, self.dtype)
        if y_val is not None:
            y_val = y_val.to(self.device, self.dtype).reshape(-1)

        mu = float(cfg.mu_init)
        best_state = self.get_param_vector().detach().clone()
        best_val_score = -1e18
        bad_epochs = 0

        stop_reason = "epochs_max"
        final_error = None
        final_grad = None
        epochs_used = 0

        for epoch in range(1, cfg.epochs_max + 1):
            epochs_used = epoch

            # current residual and Jacobian
            yhat, Jmat = self._jacobian(X_train)        # yhat(N,), J(N,P)
            r = (yhat - y_train).reshape(-1, 1)         # (N,1)

            # MSE (mean)
            mse = torch.mean((yhat - y_train) ** 2)
            final_error = float(mse.detach().cpu().item())

            # g = J^T r / N
            N = X_train.shape[0]
            g = (Jmat.t() @ r).reshape(-1) / float(N)   # (P,)
            grad_norm = torch.norm(g)
            final_grad = float(grad_norm.detach().cpu().item())

            # stopping
            if final_error <= cfg.stop_error:
                stop_reason = "stop_error"
                break
            if final_grad <= cfg.stop_grad:
                stop_reason = "stop_grad"
                break

            # H = J^T J / N
            H = (Jmat.t() @ Jmat) / float(N)            # (P,P)

            # LM step with damping tries
            w0 = self.get_param_vector()
            accepted = False
            mse0 = mse

            I = torch.eye(H.shape[0], device=self.device, dtype=self.dtype)

            for _ in range(cfg.max_damping_tries):
                mu_t = min(mu, cfg.mu_max)
                A = H + mu_t * I
                try:
                    delta = torch.linalg.solve(A, -g)   # (P,)
                except RuntimeError:
                    mu = min(mu * cfg.mu_inc, cfg.mu_max)
                    continue

                self.set_param_vector(w0 + delta)

                # evaluate new mse on train
                y_new = self.forward(X_train)
                mse_new = torch.mean((y_new - y_train) ** 2)

                if mse_new < mse0:
                    # accept
                    accepted = True
                    mu = max(mu * cfg.mu_dec, 1e-16)
                    break
                else:
                    # reject and increase mu
                    self.set_param_vector(w0)
                    mu = min(mu * cfg.mu_inc, cfg.mu_max)

            if not accepted:
                # if cannot accept a step, we stop to avoid infinite loop
                stop_reason = "damping_fail"
                break

            # ---- validation tracking (optional) ----
            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    pv = self.forward(X_val).detach().cpu().numpy()
                    yv = y_val.detach().cpu().numpy()

                # metrics
                pred = (pv > threshold).astype(np.int32)
                acc = float(np.mean(pred == yv.astype(np.int32)))

                if model_select_metric == "val_acc":
                    score = acc
                elif model_select_metric == "val_auc":
                    # robust AUC when one class missing
                    try:
                        from sklearn.metrics import roc_auc_score
                        score = float(roc_auc_score(yv.astype(int), pv))
                    except Exception:
                        score = -1e9
                else:
                    score = acc

                if score > best_val_score:
                    best_val_score = score
                    best_state = self.get_param_vector().detach().clone()
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if bad_epochs >= patience:
                    stop_reason = "early_stop_patience"
                    break

        # restore best by val (if provided)
        if X_val is not None and y_val is not None:
            self.set_param_vector(best_state)

        return {
            "epochs_used": int(epochs_used),
            "final_error": float(final_error if final_error is not None else np.nan),
            "final_grad": float(final_grad if final_grad is not None else np.nan),
            "stop_reason": str(stop_reason),
        }

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        X = X.to(self.device, self.dtype)
        with torch.no_grad():
            p = self.forward(X).detach().cpu().numpy()
        return p.astype(np.float64)

    def to_pickle_dict(self) -> dict:
        """
        CPU-friendly payload for pickle saving.
        """
        return {
            "J": int(self.J),
            "D": int(self.D),
            "W1": self.W1.detach().cpu().numpy(),
            "b1": self.b1.detach().cpu().numpy(),
            "w2": self.w2.detach().cpu().numpy(),
            "b2": self.b2.detach().cpu().numpy(),
        }

    @staticmethod
    def from_pickle_dict(d: dict, device: torch.device, dtype: torch.dtype = torch.float64) -> "LMMLP":
        D = int(d.get("D", d["W1"].shape[1]))  # backward compatible
        m = LMMLP(J=int(d["J"]), D=D, device=device, dtype=dtype, seed=0)
        m.W1 = torch.tensor(d["W1"], device=device, dtype=dtype).contiguous()
        m.b1 = torch.tensor(d["b1"], device=device, dtype=dtype).contiguous()
        m.w2 = torch.tensor(d["w2"], device=device, dtype=dtype).contiguous()
        m.b2 = torch.tensor(d["b2"], device=device, dtype=dtype).contiguous()
        return m
