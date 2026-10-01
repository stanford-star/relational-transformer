import torch


class SwaState:
    def __init__(self, named_tensors, momentum):
        self.momentum = momentum
        self.params = {name: t.detach().float().clone() for name, t in named_tensors}
        self.n = 0

    @torch.no_grad()
    def update(self, named_tensors):
        self.n += 1
        if self.momentum == 1.0:
            alpha = 1.0 / self.n
        else:
            m = self.momentum
            alpha = (1.0 - m) / (1.0 - m**self.n)
        src = dict(named_tensors)
        assert src.keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(src) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(src))}"
        )
        for name, target in self.params.items():
            target.lerp_(src[name].float(), alpha)

    def state_dict(self):
        return {
            "momentum": self.momentum,
            "n": self.n,
            "params": {k: v.detach().cpu().clone() for k, v in self.params.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        assert state["momentum"] == self.momentum, (
            f"momentum mismatch: ckpt={state['momentum']} cfg={self.momentum}"
        )
        assert state["params"].keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(state['params']) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(state['params']))}"
        )
        self.n = state["n"]
        for k, v in self.params.items():
            v.copy_(state["params"][k].to(v.device))

    @torch.no_grad()
    def sync_to(self, named_tensors):
        dst = dict(named_tensors)
        assert dst.keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(dst) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(dst))}"
        )
        for name, target in dst.items():
            target.copy_(self.params[name])
