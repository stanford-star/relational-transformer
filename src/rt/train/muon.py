import math
from collections import defaultdict

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

_DEFAULT_A = 3.4445
_DEFAULT_B = -4.7750
_DEFAULT_C = 2.0315
_DEFAULT_NS_STEPS = 5
_EPS = 1e-7


def _batched_newton_schulz(
    X: Tensor,
    steps: int,
    eps: float,
    a: float,
    b: float,
    c: float,
) -> Tensor:
    X = X.bfloat16()
    X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps))
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))
        G = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, G, X, beta=a)
    return X


def _lr_factor(adjust_lr_fn: str | None, shape: torch.Size) -> float:
    m, n = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return math.sqrt(max(1, m / n))
    elif adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(m, n))
    return 1.0


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        ns_steps: int = _DEFAULT_NS_STEPS,
        ns_coefficients: tuple[float, float, float] = (
            _DEFAULT_A,
            _DEFAULT_B,
            _DEFAULT_C,
        ),
        eps: float = _EPS,
        adjust_lr_fn: str | None = None,
        compile: bool = False,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon requires 2D parameters, got shape {p.shape}"
                    )

        self._shape_indices: list[dict[torch.Size, list[int]]] = []
        for group in self.param_groups:
            by_shape: dict[torch.Size, list[int]] = defaultdict(list)
            for idx, p in enumerate(group["params"]):
                by_shape[p.shape].append(idx)
            self._shape_indices.append(dict(by_shape))

        self._param_lr_factors: list[list[float]] = []
        for group in self.param_groups:
            self._param_lr_factors.append(
                [_lr_factor(group["adjust_lr_fn"], p.shape) for p in group["params"]]
            )

        if compile:
            self._step_impl = torch.compile(self._step_impl)

    @torch.no_grad()
    def _step_impl(self, lrs):
        for gidx, group in enumerate(self.param_groups):
            lr = lrs[gidx]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            wd = group["weight_decay"]
            a, b, c = group["ns_coefficients"]
            params = group["params"]

            active_params: list[Tensor] = []
            active_grads: list[Tensor] = []
            active_bufs: list[Tensor] = []
            active_indices: list[int] = []

            for i, p in enumerate(params):
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                active_params.append(p)
                active_grads.append(p.grad)
                active_bufs.append(state["momentum_buffer"])
                active_indices.append(i)

            if not active_params:
                continue

            torch._foreach_lerp_(active_bufs, active_grads, 1 - momentum)

            if nesterov:
                updates = torch._foreach_lerp(active_grads, active_bufs, momentum)
            else:
                updates = active_bufs

            ortho: list[Tensor | None] = [None] * len(active_params)

            active_set = {idx: j for j, idx in enumerate(active_indices)}

            for shape, param_indices in self._shape_indices[gidx].items():
                local = [active_set[i] for i in param_indices if i in active_set]
                if not local:
                    continue

                m, n = shape
                need_T = m > n

                stacked = torch.stack([updates[j] for j in local])
                if need_T:
                    stacked = stacked.transpose(-2, -1)

                X = _batched_newton_schulz(stacked, ns_steps, eps, a, b, c)

                if need_T:
                    X = X.transpose(-2, -1)

                for k, j in enumerate(local):
                    ortho[j] = X[k]

            factor_groups: dict[float, tuple[list[Tensor], list[Tensor]]] = defaultdict(
                lambda: ([], [])
            )
            for j, p in enumerate(active_params):
                if ortho[j] is None:
                    continue
                factor = self._param_lr_factors[gidx][active_indices[j]]
                p_list, o_list = factor_groups[factor]
                p_list.append(p)
                o_list.append(ortho[j])

            torch._foreach_mul_(active_params, 1 - lr * wd)

            for factor, (p_list, o_list) in factor_groups.items():
                scaled = torch._foreach_mul(o_list, -(lr * factor))
                torch._foreach_add_(p_list, scaled)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        lrs = [
            torch.tensor(
                group["lr"], dtype=torch.float64, device=group["params"][0].device
            )
            for group in self.param_groups
        ]
        self._step_impl(lrs)
        return loss
