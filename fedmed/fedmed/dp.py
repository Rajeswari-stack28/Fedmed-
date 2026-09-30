"""Differential privacy on the client update (Gaussian mechanism, DP-FedAvg style).

delta = w_local - w_global  ->  clip to L2 <= C  ->  add N(0, (sigma*C)^2)  ->  upload w_global + delta.
Noise is added BEFORE encryption, so the released global model is protected against inversion attacks.
"""
import math

import numpy as np


def privatize(local_flat, global_flat, clip, sigma, rng):
    delta = local_flat - global_flat
    norm = float(np.linalg.norm(delta))
    if clip > 0:
        delta = delta * min(1.0, clip / (norm + 1e-12))
    if sigma > 0 and clip > 0:
        delta = delta + rng.normal(0.0, sigma * clip, size=delta.shape)
    return global_flat + delta, norm


def approx_epsilon(sigma, rounds, delta=1e-5):
    """LOOSE upper bound: classic Gaussian-mechanism eps per round, basic composition over rounds.
    Client-level guarantee. Use an RDP/moments accountant (e.g. Opacus) for publishable numbers."""
    if sigma <= 0:
        return math.inf
    return rounds * math.sqrt(2 * math.log(1.25 / delta)) / sigma
