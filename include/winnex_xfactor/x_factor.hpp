/**
 * x_factor.hpp — the X factor: manifold basis of a model's embedding space.
 *
 * For ANY model, the X factor is the orthonormal basis of the manifold where
 * its token embeddings live — computed deterministically (no training) from
 * the model's embed_tokens via eigendecomposition of the covariance:
 *
 *   1. Ē = E − mean(E)                     (center)
 *   2. C = ĒᵀĒ / V                         (D×D covariance)
 *   3. C = U Λ Uᵀ                           (eigh; U orthonormal, Λ descending)
 *   4. r = min{r : cumsum(Λ₁..r)/sum(Λ) ≥ τ}   (effective rank, τ=0.95)
 *   5. X = U[:, :r]                         (the X factor: D×r manifold basis)
 *   6. P = X·Xᵀ                             (orthogonal projector onto the manifold)
 *
 * To embed text into the model's space WITHOUT training:
 *   7. ψ_c ∈ ℝ^d                            (spectral token of a character)
 *   8. ψ̃_c = expand(ψ_c) ∈ ℝ^D              (zero-pad + L2 normalize)
 *   9. h_c = P·ψ̃_c ∈ ℝ^D                    (project onto the manifold)
 *   10. h₁..hₙ → model layers → logits      (forward pass)
 *
 * The projector is idempotent (P² = P) and symmetric (Pᵀ = P), so it is the
 * orthogonal projection onto the model's embedding manifold.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_XFACTOR_X_FACTOR_HPP
#define WINNEX_XFACTOR_X_FACTOR_HPP

#include <cmath>
#include <cstdint>
#include <vector>

namespace winnex_xfactor {

/**
 * XFactor — the deterministic manifold basis of a model's embedding space.
 *
 * Computed once per model from embed_tokens. Thread-safe after construction.
 */
class XFactor {
public:
    // Computes the X factor from the model's embed_tokens (row-major, V×D).
    // Throws on invalid input.
    XFactor(const float* embed_tokens, int vocab, int dim, double variance_tau = 0.95);

    // Dimension of the input embedding space.
    int dim() const { return dim_; }

    // Effective rank r (number of principal directions kept).
    int rank() const { return (int)X_.size() / dim_; }

    // The X factor basis (D×r, column-major). Access for inspection.
    const float* basis() const { return X_.data(); }

    // The orthogonal projector P = X·Xᵀ (D×D, row-major).
    // Precomputed for fast projection.
    const float* projector() const { return P_.data(); }

    // Projects a D-dim vector onto the manifold: h' = P·h.
    // in/out are length dim(). in and out may alias.
    void project(const float* in, float* out) const;

    // Projects a batch of n vectors (n×D) onto the manifold.
    void project_batch(const float* in, float* out, int n) const;

    // Variance fraction captured by the effective rank (e.g. 0.95).
    double variance_captured() const { return variance_captured_; }

    // UB Width: the Cauchy-Schwarz residual of a vector w.r.t. the manifold
    // projector, e(h) = sqrt(||h||^2 − ||P h||^2). For a unit vector this is
    // the width of the retrieval bound: small residual = the vector lives on
    // the manifold (tight bound, prune-able); large = far off it (wide bound).
    double residual(const float* in) const {
        double n2 = 0.0, pn2 = 0.0;
        for (int b = 0; b < dim_; ++b) n2 += (double)in[b] * in[b];
        for (int a = 0; a < dim_; ++a) {
            const float* Pa = P_.data() + (size_t)a * dim_;
            double s = 0.0;
            for (int b = 0; b < dim_; ++b) s += (double)Pa[b] * in[b];
            pn2 += s * s;
        }
        double d = n2 - pn2;
        return d > 0.0 ? std::sqrt(d) : 0.0;
    }

    // Variance fraction captured by the top-k principal directions
    // (k <= dim). Useful to route the search: low-rank corpus → fold; else
    // exact scan.
    double variance_fraction(int k) const {
        if (variance_total_ <= 0.0) return 0.0;
        double acc = 0.0;
        for (int j = 0; j < k && j < (int)evals_.size(); ++j) acc += std::max(0.0, (double)evals_[j]);
        return acc / variance_total_;
    }

    // The smallest rank whose variance fraction reaches `frac` (0..1). The
    // Fold Limit rank: the minimal s such that ν(s) ≥ frac.
    int rank_for_variance(double frac) const {
        if (variance_total_ <= 0.0 || frac <= 0.0) return 0;
        double target = frac * variance_total_;
        double acc = 0.0;
        for (int j = 0; j < (int)evals_.size(); ++j) {
            acc += std::max(0.0, (double)evals_[j]);
            if (acc >= target) return j + 1;
        }
        return (int)evals_.size();
    }

    // THE ROUTER (Fold Limit, Eq. 10). Decide whether the corpus is foldable
    // by the bound engine, given a separation ratio τ_K (the K-th best exact
    // score, e.g. 0.86 for arXiv top-10) and the candidate fold rank s.
    //
    //   foldable  ⟺  ν(s)  >  1 − (τ_K / (2·E[e(q)]))²
    //
    // The spectral gap is NOT a reliable signal on real data (arXiv has
    // gap ≈ 1 yet ν(192) = 0.94 — perfectly foldable). The variance profile
    // is the correct criterion. Returns true when folding at rank s is
    // mathematically guaranteed to enable pruning (bound can fall below the
    // separation).
    bool foldable(int s, double tau_k, double mean_e_q) const {
        double e_q = mean_e_q > 0.0 ? mean_e_q : 1.0;
        double rhs = 1.0 - std::pow(tau_k / (2.0 * e_q), 2.0);
        return variance_fraction(s) > rhs;
    }

    // Mean Cauchy-Schwarz residual over a batch (unit-norm input) — the "UB
    // Width" of the whole corpus: how tightly the manifold captures it.
    double mean_residual(const float* in, int n) const {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += residual(in + (size_t)i * dim_);
        return n > 0 ? acc / n : 0.0;
    }

private:
    int dim_ = 0;
    double variance_tau_ = 0.95;
    double variance_captured_ = 0.0;
    double variance_total_ = 0.0;
    std::vector<float> evals_;  // top-r eigenvalues (for variance_fraction)
    std::vector<float> X_;  // D×r (row-major: [D][r])
    std::vector<float> P_;  // D×D (row-major: the projector)
};

/**
 * expand_spectral — expands a d-dim spectral token to D-dim with zero-pad +
 * L2 normalization (step 8 of the X-factor math).
 */
void expand_spectral(const float* psi, int d, float* out, int D);

} // namespace winnex_xfactor

#endif // WINNEX_XFACTOR_X_FACTOR_HPP
