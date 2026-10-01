"""Image quality metrics for normalized [0, 1] grayscale arrays."""
import numpy as np


def quality(reference, estimate):
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape or reference.size == 0:
        raise ValueError('Arrays must have identical, non-empty shapes')
    if not (np.isfinite(reference).all() and np.isfinite(estimate).all()):
        raise ValueError('Arrays must contain finite values')
    if min(reference.min(), estimate.min()) < 0 or max(reference.max(), estimate.max()) > 1:
        raise ValueError('Expected normalized values in [0, 1]')
    mse = float(np.mean((reference - estimate) ** 2))
    return {'mse': mse, 'psnr_db': float(-10 * np.log10(mse)) if mse else float('inf')}


def add_noise(clean, sigma, rng):
    if sigma < 0:
        raise ValueError('sigma must be non-negative')
    clean = np.asarray(clean, dtype=np.float32)
    return np.clip(clean + rng.normal(0, sigma, size=clean.shape), 0, 1).astype(np.float32)
