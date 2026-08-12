<div align="center">

# winnex-xfactor

**Deterministic manifold embedding for native LLM inference.**

The X-Factor is a training-free operator that embeds text into a pretrained
model's latent space — computed *structurally* from `embed_tokens`, never by
running the model.

[![PyPI](https://img.shields.io/badge/pip%20install-winnex--xfactor-467C45)](https://pypi.org/project/winnex-xfactor)
[![Python](https://img.shields.io/badge/Python-3.8%2B-467C45)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20-467C45)](https://isocpp.org/)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-467C45)](LICENSE)

</div>

## What it is

For any model, the X-Factor is the orthonormal basis of the manifold where its
token embeddings live:

```
Ē = E − mean(E)                  (center)
C = ĒᵀĒ / V                       (D×D covariance)
C = U Λ Uᵀ                         (eigh, Λ descending)
r = min{r : cumsum(Λ₁..r)/ΣΛ ≥ τ}  (effective rank)
X = U[:, :r]                       (the X factor: D×r manifold basis)
P = X·Xᵀ                           (orthogonal projector)
```

The projector is **idempotent** (P²=P) and **symmetric** (Pᵀ=P) — the
orthogonal projection onto the model's embedding manifold.

## Why it is fast

The eigendecomposition uses **power iteration with deflation**:

```
O(D²·r·iters)  vs  O(D³) for a full Jacobi/eigh
```

~2000× faster at D=2048, with the same deterministic guarantee (the Madhava
philosophy: bound-guided selection, provable operators, no black boxes).

## Install

```bash
pip install winnex-xfactor
```

Requires Python ≥ 3.8 and NumPy. The C++ core ships pre-built in the wheel.

## Quick start

```python
import numpy as np
import winnex_xfactor

# 1. Compute the X-Factor from a model's embed_tokens (a sample is enough).
E = np.random.randn(5000, 2048).astype(np.float32)
P, rank, variance = winnex_xfactor.compute_xfactor(E, 5000, 2048, tau=0.95)
print(f"rank={rank}, variance captured={variance:.3f}")

# 2. Embed a spectral token into the model's manifold.
psi = np.ones(64, dtype=np.float32)
h = winnex_xfactor.embed_spectral(psi, P)   # length 2048, on the manifold
```

## API

- `compute_xfactor(embed_tokens, vocab, dim, tau=0.95) -> (P, rank, variance)`
- `XFactor(embed_tokens, vocab, dim, tau)` — object with `.project`, `.project_batch`,
  `.projector`, `.basis`, `.rank`, `.variance_captured`, `.dim`
- `expand_spectral(psi, d, D)` — zero-pad + L2 normalize
- `embed_spectral(psi, P)` — full text→manifold projection

## License

BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
