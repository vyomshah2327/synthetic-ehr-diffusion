import torch
import copy


class EMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy updated after every optimizer step.
    Use apply_shadow() before evaluation, restore() to resume training.
    """

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow.state_dict())

    def restore(self, model):
        model.load_state_dict(self._backup)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)
