"""Neural network model for Azul AlphaZero-style pipeline.

Architecture:
  Input: state vector (float32) size 158
  Hidden: 3 x 256-dim fully connected with ReLU
  Heads:
    - Policy: 180 logits (before masking)
    - Value: scalar in [-1,1] via tanh

Interface:
  model = AzulNet()
  policy_logits, value = model(states_tensor)  # states: (B,158)
  policy_logits, value = model.predict(state_batch_np, mask_batch_np)  # numpy convenience

Notes:
  - predict() returns raw policy logits (no illegal action masking). Caller should apply mask:
        masked_logits = logits + (mask <= 0) * -1e9
  - save/load helpers provided.
"""
from __future__ import annotations
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

INPUT_DIM = 158
POLICY_DIM = 180
HIDDEN_DIM = 256
HIDDEN_LAYERS = 3

class AzulNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, policy_dim: int = POLICY_DIM,
                 hidden_dim: int = HIDDEN_DIM, hidden_layers: int = HIDDEN_LAYERS,
                 use_layernorm: bool = False):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, policy_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Xavier init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, INPUT_DIM) float32 tensor
        Returns: (policy_logits (B,180), value (B,1))"""
        h = self.backbone(x)
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))  # value in [-1,1]
        return policy_logits, value

    @torch.no_grad()
    def predict(self, states: Union[np.ndarray, torch.Tensor],
                masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
                device: Optional[torch.device] = None,
                apply_mask: bool = False,
                mask_fill: float = -1e9) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience inference.
        states: (B,158) numpy or tensor
        masks:  optional (B,180) 0/1 for legal actions
        If apply_mask=True and masks provided -> illegal logits filled with mask_fill.
        Returns numpy arrays: (policy_logits, values_flat)
        """
        if device is None:
            device = next(self.parameters()).device
        if isinstance(states, np.ndarray):
            st = torch.from_numpy(states).float().to(device)
        else:
            st = states.to(device)
        policy_logits, value = self.forward(st)
        if masks is not None and apply_mask:
            if isinstance(masks, np.ndarray):
                m = torch.from_numpy(masks).to(device)
            else:
                m = masks.to(device)
            policy_logits = policy_logits + (m <= 0) * mask_fill
        return policy_logits.cpu().numpy(), value.squeeze(-1).cpu().numpy()

    def save(self, path: str):
        torch.save({'model_state': self.state_dict(),
                    'config': {
                        'input_dim': INPUT_DIM,
                        'policy_dim': POLICY_DIM,
                        'hidden_dim': HIDDEN_DIM,
                        'hidden_layers': HIDDEN_LAYERS,
                    }}, path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> 'AzulNet':
        blob = torch.load(path, map_location=map_location)
        cfg = blob.get('config', {})
        model = AzulNet(**{k: cfg.get(k, globals()[k.upper()]) for k in ['input_dim','policy_dim','hidden_dim','hidden_layers']})
        model.load_state_dict(blob['model_state'])
        return model

# Utility function for masking outside class (optional)

def apply_policy_mask(logits: torch.Tensor, mask: torch.Tensor, fill: float = -1e9) -> torch.Tensor:
    """Return masked logits (does not modify in-place). mask shape (B, POLICY_DIM)."""
    return logits + (mask <= 0) * fill

__all__ = [
    'AzulNet', 'apply_policy_mask', 'INPUT_DIM', 'POLICY_DIM'
]
