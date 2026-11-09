"""Signal/context fusion experiment; inputs use HW2 train-only standardization."""
import torch
from torch import nn
from training.model import ActivationMLP


class DualBranch(nn.Module):
    def __init__(self, gated=False, context_dropout=0.0, signal_skip=False):
        super().__init__()
        self.signal = nn.Sequential(nn.Linear(40, 32), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        # Initialize shared layers before the extra gate so the ablation and
        # gated model start with identical branch/classifier weights per seed.
        self.gate = nn.Linear(64, 32) if gated else None
        if self.gate is not None:
            # Start at equal weighting, while allowing gate gradients immediately.
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.context_dropout = float(context_dropout)
        self.signal_skip = nn.Linear(40, 2) if signal_skip else None
        if self.signal_skip is not None:
            nn.init.zeros_(self.signal_skip.weight)
            nn.init.zeros_(self.signal_skip.bias)

    def features(self, x):
        signal, context = self.signal(x[:, :40]), self.context(x[:, 40:])
        if self.training and self.context_dropout:
            # Drop the entire context branch per sample, not isolated coordinates.
            # Inverted scaling preserves its expected magnitude.
            keep = (torch.rand((x.shape[0], 1), device=x.device) >= self.context_dropout).to(x.dtype)
            context = context * keep / (1 - self.context_dropout)
        gate = torch.sigmoid(self.gate(torch.cat([signal, context], dim=1))) if self.gate is not None else None
        if gate is not None:
            # Factor two makes gate=0.5 identical in scale to direct concatenation.
            signal, context = 2 * gate * signal, 2 * (1 - gate) * context
        return torch.cat([signal, context], dim=1), gate

    def forward(self, x):
        logits = self.classifier(self.features(x)[0])
        if self.signal_skip is not None:
            logits = logits + self.signal_skip(x[:, :40])
        return logits


def make_model(name, settings=None):
    settings = settings or {}
    if name == "baseline":
        return ActivationMLP(44)
    if name == "dual_branch":
        return DualBranch(False)
    if name == "gated_dual_branch":
        return DualBranch(True)
    if name in {"gated_decay", "gated_context25", "gated_context50", "gated_context25_skip"}:
        return DualBranch(True, settings.get("contextDropout", 0.0), settings.get("signalSkip", False))
    raise ValueError(name)
