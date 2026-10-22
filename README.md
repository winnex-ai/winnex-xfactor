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

**Honest cost (corrected 2026-10-21).** An earlier version of this README
claimed "~2000× faster at D=2048". That number does not reproduce, and the sign
is wrong at the dimension it names. Measured against `numpy.eigh` on the same
covariance:

| D | power iteration | numpy eigh | speedup |
|---|---|---|---|
| 128 | 0.25 s | 0.43 s | 1.7× |
| 384 | 7.4 s | 8.9 s | 1.2× |
| **2048** | **399 s** | **34.7 s** | **0.09× (11× slower)** |

The cause is structural: `max_candidates = D` with up to **2000 iterations per
component**. At D=2048 the effective rank is ~1656, so the cost is
1656 × 2000 × O(D²) against a single O(D³) `eigh`.

What the method actually wins is the **low-rank regime** — a corpus whose
effective rank is small relative to D. Measured on genuinely low-rank
embeddings (rank 40, in D=512 and D=1024): **8.3×** and **5.6×** faster than
`eigh`.

**Recommendation.** Use this operator when the manifold is known to be
low-rank and you want a deterministic, dependency-free implementation. For
D ≥ 2048 with a high effective rank, `numpy.eigh` — or a randomized SVD
(Halko–Martinsson–Tropp) — is an order of magnitude faster and should be
preferred.

The mathematical output is correct in every regime: the projector is idempotent
(‖P²−P‖/‖P‖ = 3.3e-7), exactly symmetric, and the reported rank matches the
true eigendecomposition. Only the performance claim was wrong.

The guarantee that does hold everywhere is determinism: fixed seed, fixed
iteration count, same input → same basis.

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

## Benchmark

Run the benchmark on Kaggle — installs `winnex-xfactor` (and `winnex-nano` for
the safetensors loader) from PyPI and computes the X-Factor from **real
embeddings** of a Qwen model (Kaggle `qwen-lm/qwen`):

[![Kaggle](https://img.shields.io/badge/Kaggle-Manifold%20benchmark-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-xfactor-benchmark-manifold)

Result: effective rank 118 (D=128, τ=0.95, variance captured 0.954),
compute 0.03s — power iteration with deflation (O(D²·r)) on real embeddings.

## License

BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
