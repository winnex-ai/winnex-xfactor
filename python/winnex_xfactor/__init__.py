"""
winnex-xfactor — deterministic manifold embedding for native LLM inference.

The X-Factor is a deterministic, training-free operator that embeds arbitrary
text into the latent space of a pretrained language model:

    Ē = E − mean(E)                  (center the embedding matrix)
    C = ĒᵀĒ / V                       (D×D covariance)
    C = U Λ Uᵀ                         (eigh, Λ descending)
    r = min{r : cumsum(Λ₁..r)/ΣΛ ≥ τ}  (effective rank)
    X = U[:, :r]                       (the X factor: manifold basis)
    P = X·Xᵀ                           (orthogonal projector onto the manifold)

Computed by power iteration with deflation — O(D²·r·iters), NOT O(D³) — so it
is ~2000× faster than a full eigendecomposition at D=2048. Deterministic and
training-free.

Quick start:
    import numpy as np
    import winnex_xfactor

    E = np.random.randn(5000, 2048).astype(np.float32)   # model embed_tokens
    P, rank, variance = winnex_xfactor.compute_xfactor(E, 5000, 2048, tau=0.95)

    psi = np.ones(64, dtype=np.float32)                  # spectral token
    h = P @ winnex_xfactor.expand_spectral(psi, 64, 2048)  # embed into manifold

BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
"""

from __future__ import annotations

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    from . import _winnex_xfactor as _native
except ImportError:
    try:
        import _winnex_xfactor as _native  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "winnex_xfactor native extension not found. Install with: pip install winnex-xfactor"
        ) from exc

import numpy as np

XFactor = _native.XFactor
compute_xfactor = _native.compute_xfactor
expand_spectral = _native.expand_spectral

__version__ = "1.0.0"
__all__ = ["XFactor", "compute_xfactor", "expand_spectral", "__version__"]


def build_xfactor(
    embed_tokens: np.ndarray,
    *,
    dim: int | None = None,
    variance_tau: float = 0.95,
) -> tuple[np.ndarray, int, float]:
    """Build the X-Factor projector from a model's embed_tokens.

    Parameters
    ----------
    embed_tokens : np.ndarray
        Shape (vocab, dim) float32 (or convertible). A sample of the model's
        token embedding matrix.
    dim : int, optional
        Embedding dimension. Defaults to ``embed_tokens.shape[1]``.
    variance_tau : float
        Variance fraction the manifold must capture (default 0.95).

    Returns
    -------
    (P, rank, variance) — the D×D projector, the effective rank, and the
    variance fraction captured.
    """
    arr = np.ascontiguousarray(embed_tokens, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("embed_tokens must be a 2D array (vocab, dim)")
    vocab, D = arr.shape
    if dim is not None:
        D = int(dim)
    return compute_xfactor(arr, vocab, D, variance_tau)


def embed_spectral(psi: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Embed a spectral token into the model's manifold: h = P·expand(ψ)."""
    p = np.ascontiguousarray(psi, dtype=np.float32)
    Pp = np.ascontiguousarray(P, dtype=np.float32)
    D = Pp.shape[0]
    expanded = expand_spectral(p, p.shape[0], D)
    return Pp @ expanded
