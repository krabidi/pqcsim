#pragma once

// Quantum linear-optics simulation in the Fock (photon-number) basis.
//
// A FockState is a superposition  sum_n c_n |n_0, n_1, ..., n_{M-1}>  over M optical modes.
// Components act on creation operators:  a_in^dagger -> sum_out U[out][in] a_out^dagger.

#include <cmath>
#include <complex>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pqc {

using cplx = std::complex<double>;
using Occupation = std::vector<int>;

namespace fock_detail {

inline double factorial(int n) {
    return std::tgamma(n + 1.0);
}

inline double binomial(int n, int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

}  // namespace fock_detail

class FockState {
private:
    int n_modes_;
    std::map<Occupation, cplx> amps_;

    void check_mode(int m) const {
        if (m < 0 || m >= n_modes_) {
            throw std::out_of_range("mode index " + std::to_string(m) + " out of range");
        }
    }

public:
    static constexpr double kTol = 1e-14;

    // Start in the basis state |occupation>.
    explicit FockState(const Occupation& occupation) : n_modes_(static_cast<int>(occupation.size())) {
        if (n_modes_ == 0) {
            throw std::invalid_argument("FockState needs at least one mode");
        }
        for (int n : occupation) {
            if (n < 0) {
                throw std::invalid_argument("photon numbers must be non-negative");
            }
        }
        amps_[occupation] = 1.0;
    }

    int n_modes() const { return n_modes_; }

    const std::map<Occupation, cplx>& terms() const { return amps_; }

    cplx amplitude(const Occupation& occupation) const {
        auto it = amps_.find(occupation);
        return it == amps_.end() ? cplx(0.0) : it->second;
    }

    double probability(const Occupation& occupation) const {
        return std::norm(amplitude(occupation));
    }

    double norm() const {
        double total = 0.0;
        for (const auto& [occ, a] : amps_) total += std::norm(a);
        return total;
    }

    double mean_photon_number(int mode) const {
        check_mode(mode);
        double total = 0.0;
        for (const auto& [occ, a] : amps_) total += occ[mode] * std::norm(a);
        return total;
    }

    // Two-mode unitary on modes (i, j); u = {{U00, U01}, {U10, U11}} with U[out][in].
    void apply_two_mode_unitary(int i, int j, const cplx u[2][2]) {
        check_mode(i);
        check_mode(j);
        if (i == j) throw std::invalid_argument("two-mode unitary needs two distinct modes");

        using fock_detail::binomial;
        using fock_detail::factorial;

        std::map<Occupation, cplx> out;
        for (const auto& [occ, a] : amps_) {
            const int ni = occ[i];
            const int nj = occ[j];
            // (a_i^dag)^ni (a_j^dag)^nj / sqrt(ni! nj!), each operator expanded binomially.
            const double pre = 1.0 / std::sqrt(factorial(ni) * factorial(nj));
            for (int k = 0; k <= ni; ++k) {        // photons from i that stay in i
                for (int l = 0; l <= nj; ++l) {    // photons from j that move to i
                    const int mi = k + l;
                    const int mj = ni + nj - mi;
                    cplx c = a * pre * binomial(ni, k) * binomial(nj, l) *
                             std::pow(u[0][0], k) * std::pow(u[1][0], ni - k) *
                             std::pow(u[0][1], l) * std::pow(u[1][1], nj - l) *
                             std::sqrt(factorial(mi) * factorial(mj));
                    Occupation next = occ;
                    next[i] = mi;
                    next[j] = mj;
                    out[next] += c;
                }
            }
        }
        amps_.clear();
        for (const auto& [occ, a] : out) {
            if (std::abs(a) > kTol) amps_[occ] = a;
        }
    }

    // Beamsplitter with transmission cos(theta); theta = pi/4 is 50:50.
    // U = [[cos t, i e^{-i phi} sin t], [i e^{i phi} sin t, cos t]]
    void apply_beamsplitter(int i, int j, double theta, double phi = 0.0) {
        const cplx I(0.0, 1.0);
        const cplx u[2][2] = {
            {std::cos(theta), I * std::exp(-I * phi) * std::sin(theta)},
            {I * std::exp(I * phi) * std::sin(theta), std::cos(theta)},
        };
        apply_two_mode_unitary(i, j, u);
    }

    // a^dag -> e^{i phi} a^dag on one mode, so |n> picks up e^{i n phi}.
    void apply_phase_shift(int mode, double phi) {
        check_mode(mode);
        for (auto& [occ, a] : amps_) {
            a *= std::exp(cplx(0.0, occ[mode] * phi));
        }
    }

    // Draw photon-number-resolving detection outcomes on all modes.
    std::vector<Occupation> sample(int shots, unsigned seed) const {
        std::vector<Occupation> outcomes;
        std::vector<double> weights;
        for (const auto& [occ, a] : amps_) {
            outcomes.push_back(occ);
            weights.push_back(std::norm(a));
        }
        std::mt19937_64 rng(seed);
        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        std::vector<Occupation> result;
        result.reserve(shots);
        for (int s = 0; s < shots; ++s) result.push_back(outcomes[dist(rng)]);
        return result;
    }
};

// An ordered list of linear-optical operations that can be applied to any FockState.
class OpticalCircuit {
private:
    struct Op {
        enum Kind { BEAMSPLITTER, PHASE } kind;
        int i;
        int j;
        double theta;
        double phi;
    };

    int n_modes_;
    std::vector<Op> ops_;

    void check_mode(int m) const {
        if (m < 0 || m >= n_modes_) {
            throw std::out_of_range("mode index " + std::to_string(m) + " out of range");
        }
    }

public:
    explicit OpticalCircuit(int n_modes) : n_modes_(n_modes) {
        if (n_modes < 1) throw std::invalid_argument("OpticalCircuit needs at least one mode");
    }

    int n_modes() const { return n_modes_; }
    size_t size() const { return ops_.size(); }

    OpticalCircuit& beamsplitter(int i, int j, double theta = M_PI / 4, double phi = 0.0) {
        check_mode(i);
        check_mode(j);
        if (i == j) throw std::invalid_argument("beamsplitter needs two distinct modes");
        ops_.push_back({Op::BEAMSPLITTER, i, j, theta, phi});
        return *this;
    }

    OpticalCircuit& phase_shift(int mode, double phi) {
        check_mode(mode);
        ops_.push_back({Op::PHASE, mode, -1, 0.0, phi});
        return *this;
    }

    FockState run(FockState state) const {
        if (state.n_modes() != n_modes_) {
            throw std::invalid_argument("state has " + std::to_string(state.n_modes()) +
                                        " modes but circuit has " + std::to_string(n_modes_));
        }
        for (const auto& op : ops_) {
            if (op.kind == Op::BEAMSPLITTER) {
                state.apply_beamsplitter(op.i, op.j, op.theta, op.phi);
            } else {
                state.apply_phase_shift(op.i, op.phi);
            }
        }
        return state;
    }
};

}  // namespace pqc
