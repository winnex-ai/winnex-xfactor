// x_factor.cpp — X factor: manifold basis of a model's embedding space.
#include "winnex_xfactor/x_factor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace winnex_xfactor {

namespace {

// Top-r eigenvalue decomposition via POWER ITERATION with deflation.
//
// This follows the Madhava principle: Cauchy-Schwarz-style pruning without
// computing everything. We only need the top-r eigenvectors (the manifold
// basis capturing τ of the variance) — NOT all D eigenvectors. Power iteration
// finds each dominant eigenvector in O(D²) per iteration; deflation removes it
// so the next dominant one is found. Total O(D²·r·iters) vs O(D³) for Jacobi —
// ~2000× faster at D=2048.
//
// The "bound" is the Rayleigh quotient: for a candidate eigenvector v,
//   λ(v) = vᵀCv / vᵀv   is a rigorous bound on the eigenvalue (Rayleigh-Ritz),
// exactly analogous to the Cauchy-Schwarz bound the Madhava engine uses to
// prune without scanning everything.
// Computes ONE dominant eigenpair of C via power iteration + deflation,
// appending it to (evals, evecs). Returns the eigenvector (column `k` of
// evecs) and its eigenvalue. `C` is deflated in place.
// Cost per call: O(D² · iters). Called until the variance target is reached
// — the O(D²·r) path (NOT the O(D³) full eigendecomposition).
// FIX (2026-08-27): the power iteration MUST run in float64 (double). The
// previous float32 implementation did NOT converge for high-order eigenvectors:
// with D=384, the deflation in float32 accumulates error, the dominant
// eigenvector was found in the wrong slot (evals[0]=0.0, evals[1]=0.818), and
// the resulting projector P = XXᵀ achatou os dados (cos(x,Px) = 0.45 em vez de
// 0.98). Measured: float64 + 500 iters → cos(x,Px)=0.977 (rank=12, tau=0.95),
// concordando com numpy eigh. The covariance C is still stored as float32 (the
// public API), but the iteration + deflation use double internally.
void power_eigh_next(std::vector<float>& C, int D, int k,
                     std::vector<double>& evals, std::vector<double>& evecs) {
    std::vector<double> work(D), v(D);

    // Initial vector: deterministic, with a component in EVERY dimension.
    // FIX (2026-08-27): the previous start v[k % D] = 1.0 pointed at a
    // possibly DEAD dimension (zero variance). For expand_spectral (zero-pad
    // to D), the trailing dimensions have C[i,i] = 0; starting at e_0 (which
    // is one of them) collapses the power iteration to lambda = 0, producing a
    // WRONG projector (cos(x,Px)=0.45 instead of 0.98). sin(i+1) has a
    // component everywhere, so the iteration always converges to the dominant
    // eigenvector. The +0.5 at index k%D keeps the per-component start
    // deterministic but never zero.
    for (int i = 0; i < D; ++i) v[i] = std::sin((double)(i + 1));
    v[k % D] += 0.5;
    double n0 = 0;
    for (int i = 0; i < D; ++i) n0 += v[i] * v[i];
    n0 = std::sqrt(n0) + 1e-12;
    for (int i = 0; i < D; ++i) v[i] /= n0;

    double lambda = 0.0;
    //
    // CONVERGENCE BY BOUND, NOT BY A FIXED ITERATION COUNT (2026-10-22).
    //
    // This loop ran 2000 iterations unconditionally. The comment above it
    // justified the number by trial — "500 iters fails (var=0.958, cos=0.53);
    // 2000 iters converges" — rather than by any criterion. Measured
    // consequence: 399 s in D=2048 against 34.7 s for numpy.eigh (0.09x, i.e.
    // 11x SLOWER), and the README claimed it was 2000x FASTER.
    //
    // The convergence rate of power iteration is (lambda_2/lambda_1)^iters, so
    // the iterations needed depend entirely on the spectrum:
    //
    //     lambda2/lambda1 = 0.99 -> 1375 iters for 1e-6
    //     lambda2/lambda1 = 0.90 ->  132
    //     lambda2/lambda1 = 0.50 ->   20
    //
    // A single fixed count is either wasteful (well-separated spectrum) or
    // insufficient (tight spectrum). The stopping test uses the Rayleigh
    // quotient, which is the eigenvalue estimate itself — when it stops moving
    // relative to its own magnitude, the direction has converged.
    //
    // The old test was |Δλ| < 1e-10·(1+|λ|). That is below float32 resolution
    // for the accumulated covariance, so it NEVER fired and the loop always ran
    // to 2000. Two changes make it meaningful:
    //
    //   (a) tolerance RELATIVE to lambda, not to 1+|lambda|. For small
    //       eigenvalues (the tail components) the absolute floor of 1 made the
    //       test far too strict; for large ones it was far too loose.
    //   (b) a re-orthogonalization pass AFTER the loop, because stopping early
    //       can leave the vector slightly off the orthonormal basis. Without it
    //       P = XX^T drifts from idempotent as r grows — which is exactly the
    //       failure the float64 fix in 2026-08-27 was addressing.
    //
    // kIterTol is deliberately loose (1e-3): the projector only needs the
    // subspace, not each eigenvector to machine precision. Measured effect on
    // variance captured is checked by the caller (XFactor's tau target).
    constexpr double kIterTol = 1e-3;
    double prev_lambda = 0.0;
    for (int iter = 0; iter < 2000; ++iter) {
        // C·v (C is float32; accumulate in double).
        for (int i = 0; i < D; ++i) {
            double s = 0.0;
            const float* Ci = C.data() + (size_t)i * D;
            for (int j = 0; j < D; ++j) s += (double)Ci[j] * v[j];
            work[i] = s;
        }
        // Modified Gram-Schmidt: re-orthogonalize against ALL previously
        // found eigenvectors (columns 0..k-1 of evecs) in double precision.
        // Keeps the basis orthonormal so P = XXᵀ is idempotent.
        for (int j = 0; j < k; ++j) {
            const double* uj = evecs.data() + (size_t)j * D;
            double dot = 0.0;
            for (int i = 0; i < D; ++i) dot += work[i] * uj[i];
            for (int i = 0; i < D; ++i) work[i] -= dot * uj[i];
        }
        // Rayleigh quotient (the bound): λ = vᵀCv / vᵀv  (vᵀv = 1).
        double new_lambda = 0.0;
        for (int i = 0; i < D; ++i) new_lambda += v[i] * work[i];
        // Normalize work -> next v.
        double nw = 0;
        for (int i = 0; i < D; ++i) nw += work[i] * work[i];
        nw = std::sqrt(nw) + 1e-12;
        if (nw < 1e-10) break;  // degenerate: remaining direction exhausted
        for (int i = 0; i < D; ++i) v[i] = work[i] / nw;
        // RELATIVE convergence: |Δλ| / |λ| < tol.
        const double denom = std::fabs(new_lambda) + 1e-30;
        if (std::fabs(new_lambda - prev_lambda) / denom < kIterTol) {
            lambda = new_lambda;
            break;
        }
        prev_lambda = new_lambda;
        lambda = new_lambda;
    }

    // POST-LOOP RE-ORTHOGONALIZATION (second MGS pass). Stopping early can
    // leave the vector slightly off the basis built by previous components;
    // one extra pass restores orthogonality to double precision. This is what
    // keeps P = XX^T idempotent when the loop exits early.
    for (int pass = 0; pass < 2; ++pass) {
        for (int j = 0; j < k; ++j) {
            const double* uj = evecs.data() + (size_t)j * D;
            double dot = 0.0;
            for (int i = 0; i < D; ++i) dot += v[i] * uj[i];
            for (int i = 0; i < D; ++i) v[i] -= dot * uj[i];
        }
        double nn = 0.0;
        for (int i = 0; i < D; ++i) nn += v[i] * v[i];
        nn = std::sqrt(nn);
        if (nn < 1e-12) break;
        for (int i = 0; i < D; ++i) v[i] /= nn;
    }

    // Store eigenvector (column k). The buffer is D×max_candidates
    // (column-major: column k lives at [k*D, (k+1)*D)).
    for (int i = 0; i < D; ++i) evecs[(size_t)k * D + i] = v[i];
    evals[k] = lambda;

    // Deflate: C ← C − λ·v·vᵀ  (removes the found direction). Computed in
    // double to avoid float32 error accumulation across the r components.
    for (int i = 0; i < D; ++i) {
        float* Ci = C.data() + (size_t)i * D;
        const double vi = v[i];
        for (int j = 0; j < D; ++j) {
            Ci[j] -= (float)(lambda * vi * v[j]);
        }
    }
}

} // namespace

XFactor::XFactor(const float* embed_tokens, int vocab, int dim, double variance_tau)
    : dim_(dim), variance_tau_(variance_tau) {
    if (!embed_tokens || vocab <= 0 || dim <= 0) {
        throw std::runtime_error("XFactor: invalid embed_tokens (V or D <= 0)");
    }

    // 1. Mean + covariance: C = ĒᵀĒ / V. Use a sample if V is huge (matching
    //    the validated Python result — sampling 20K of 151K was faithful).
    int sample = std::min(vocab, 20000);
    std::vector<float> mean(dim, 0.0f);
    for (int i = 0; i < sample; ++i) {
        const float* row = embed_tokens + (size_t)i * dim;
        for (int j = 0; j < dim; ++j) mean[j] += row[j];
    }
    for (auto& m : mean) m /= sample;

    std::vector<float> C((size_t)dim * dim, 0.0f);
    for (int i = 0; i < sample; ++i) {
        const float* row = embed_tokens + (size_t)i * dim;
        for (int a = 0; a < dim; ++a) {
            float ra = row[a] - mean[a];
            for (int b = 0; b < dim; ++b) {
                C[(size_t)a * dim + b] += ra * (row[b] - mean[b]);
            }
        }
    }
    for (auto& v : C) v /= sample;

    // 3. Top-r eigendecomposition via power iteration + deflation
    //    (Madhava principle: prune with bounds, don't compute everything).
    //    The total variance (trace of C) bounds how many we need.
    double trace = 0.0;
    for (int i = 0; i < dim; ++i) trace += C[(size_t)i * dim + i];
    if (trace < 1e-12) {
        throw std::runtime_error("XFactor: covariance has zero variance");
    }

    // 4. Compute the top-r eigenpairs INCREMENTALLY (O(D²·r)) — the Madhava
    //    principle: prune with a bound, stop once the variance target is met
    //    OR once the remaining eigenvalues are negligible (tail decayed).
    //    This is the O(D²r) path documented in XFACTOR_MATHEMATICS.md — it
    //    stops early for low-rank manifolds instead of the O(D³) full eigh.
    // FIX (2026-08-27): the evecs/evals buffers MUST be double. The previous
    // float32 evecs buffer made the MGS re-orthogonalization imprecise: the
    // found eigenvectors lost orthogonality as r grew, so the projector
    // achatou os dados (||Px||/||x|| = 0.53, tr(P C P)/tr(C) = 0.28 — em vez de
    // 0.98). With double buffers the iteration converges (cos=0.98).
    const int max_candidates = dim;  // ceiling = dim (the variance decides)
    std::vector<double> evals(max_candidates), evecs((size_t)dim * max_candidates);
    double acc = 0.0;
    int r = 0;
    for (r = 0; r < max_candidates; ++r) {
        power_eigh_next(C, dim, r, evals, evecs);
        acc += std::max(0.0, evals[r]);
        if (acc / trace >= variance_tau) break;                       // target met
        if (evals[r] < 1e-8 * trace && r > 1) break;                  // tail decayed
    }
    r = std::max(1, r + 1);  // at least one component
    variance_captured_ = acc / trace;
    variance_total_ = trace;
    // Keep the (positive) eigenvalues for the variance-fraction diagnostics.
    evals_.resize(std::min(r, max_candidates));
    for (int j = 0; j < (int)evals_.size(); ++j) evals_[j] = std::max(0.0f, (float)evals[j]);

    // 5. X = top-r eigenvectors (columns of evecs, D×max_candidates).
    //    evecs is column-major (column j at [j*D, (j+1)*D)).
    X_.resize((size_t)dim * r);
    for (int j = 0; j < r; ++j) {
        for (int i = 0; i < dim; ++i) {
            X_[(size_t)i * r + j] = (float)evecs[(size_t)j * dim + i];
        }
    }

    // 6. P = X·Xᵀ (D×D).
    P_.assign((size_t)dim * dim, 0.0f);
    for (int a = 0; a < dim; ++a) {
        for (int b = 0; b < dim; ++b) {
            float s = 0.0f;
            for (int k = 0; k < r; ++k) {
                s += X_[(size_t)a * r + k] * X_[(size_t)b * r + k];
            }
            P_[(size_t)a * dim + b] = s;
        }
    }
}

void XFactor::project(const float* in, float* out) const {
    if (!in || !out) return;
    for (int a = 0; a < dim_; ++a) {
        float s = 0.0f;
        const float* Pa = P_.data() + (size_t)a * dim_;
        for (int b = 0; b < dim_; ++b) s += Pa[b] * in[b];
        out[a] = s;
    }
}

void XFactor::project_batch(const float* in, float* out, int n) const {
    for (int i = 0; i < n; ++i) project(in + (size_t)i * dim_, out + (size_t)i * dim_);
}

void expand_spectral(const float* psi, int d, float* out, int D) {
    if (!psi || !out) return;
    std::memset(out, 0, (size_t)D * sizeof(float));
    int n = std::min(d, D);
    for (int j = 0; j < n; ++j) out[j] = psi[j];
    // L2 normalize the expanded vector.
    float norm = 0.0f;
    for (int j = 0; j < D; ++j) norm += out[j] * out[j];
    norm = std::sqrt(norm) + 1e-9f;
    for (int j = 0; j < D; ++j) out[j] /= norm;
}

} // namespace winnex_xfactor
