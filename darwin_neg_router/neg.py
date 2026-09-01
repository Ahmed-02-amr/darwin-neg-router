from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class NEGStats:
    steps: int = 0
    activations: int = 0
    entropy_sum: float = 0.0
    entropy_max: float = 0.0

    @property
    def activation_rate(self) -> float:
        return self.activations / self.steps if self.steps else 0.0

    @property
    def entropy_mean(self) -> float:
        return self.entropy_sum / self.steps if self.steps else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "activations": self.activations,
            "activation_rate": self.activation_rate,
            "entropy_mean": self.entropy_mean,
            "entropy_max": self.entropy_max,
        }


class NEGEntropyMonitor:
    """Attach the released NEG entropy head without retaining all layer states.

    The upstream helper forces ``output_hidden_states=True`` and masks to top-k.
    Top-k masking cannot change greedy argmax. This monitor instead captures the
    lm_head input with a pre-hook and exposes the learned uncertainty signal to
    the request-level candidate router. For sampled candidates, ``apply_gate``
    also applies the released top-k and temperature values safely.
    """

    def __init__(self, model: Any, weights_path: str | Path):
        import torch
        import torch.nn as nn

        from safetensors.torch import load_file

        state = load_file(str(weights_path))
        hidden = state["head.proj_down.weight"].shape[1]
        reduced = state["head.proj_down.weight"].shape[0]
        self.head = nn.Sequential(
            nn.Linear(hidden, reduced),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(reduced, 1),
            nn.Softplus(),
        )
        self.head[0].weight.data.copy_(state["head.proj_down.weight"])
        self.head[0].bias.data.copy_(state["head.proj_down.bias"])
        self.head[3].weight.data.copy_(state["head.proj_out.weight"])
        self.head[3].bias.data.copy_(state["head.proj_out.bias"])
        self.threshold = float(state["gate.threshold"].item())
        self.temperature_scale = float(state.get("gate.temp_scale", torch.tensor(1.0)).item())
        self.top_k = 20
        self.stats = NEGStats()
        self.last_entropy = None

        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            raise ValueError("The model has no lm_head to monitor")
        device = next(lm_head.parameters()).device
        self.head.to(device=device, dtype=torch.float32).eval()
        self._hook = lm_head.register_forward_pre_hook(self._capture)

    def _capture(self, _module: Any, args: tuple[Any, ...]) -> None:
        import torch

        if not args:
            return
        hidden = args[0][:, -1].float()
        with torch.inference_mode():
            self.last_entropy = self.head(hidden).squeeze(-1)

    def reset(self) -> None:
        self.stats = NEGStats()
        self.last_entropy = None

    def apply_gate(self, scores: Any, *, sampled: bool) -> Any:
        """Record activation and restrict uncertain sampled steps.

        Greedy scores are returned unchanged because the released scalar gate
        has no information with which to rerank the existing top-1 token.
        """
        import torch

        if self.last_entropy is None:
            return scores
        entropy = self.last_entropy.to(scores.device)
        active = entropy > self.threshold
        self.stats.steps += int(entropy.numel())
        self.stats.activations += int(active.sum().item())
        self.stats.entropy_sum += float(entropy.sum().item())
        self.stats.entropy_max = max(self.stats.entropy_max, float(entropy.max().item()))
        if not sampled or not bool(active.any()):
            return scores

        gated = scores.clone()
        rows = active.nonzero(as_tuple=True)[0]
        selected = gated[rows] / max(self.temperature_scale, 1e-4)
        values, indices = selected.topk(min(self.top_k, selected.shape[-1]), dim=-1)
        masked = torch.full_like(selected, float("-inf"))
        masked.scatter_(-1, indices, values)
        gated[rows] = masked
        return gated


class NEGLogitsProcessor:
    def __init__(self, monitor: NEGEntropyMonitor, *, sampled: bool):
        self.monitor = monitor
        self.sampled = sampled

    def __call__(self, _input_ids: Any, scores: Any) -> Any:
        return self.monitor.apply_gate(scores, sampled=self.sampled)

