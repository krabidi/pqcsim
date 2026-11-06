#pragma once

// Matrix functions behind multiphoton statistics.
//   permanent: Ryser's formula with Gray-code updates, O(2^n n)
//   hafnian:   dynamic programming over index subsets (always pair the lowest free index), O(2^n n)
//              memory 2^n complex numbers, so n <= 26.

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pqc {

using cplx_t = std::complex<double>;

inline cplx_t permanent_ryser(const std::vector<cplx_t>& a, int n) {
    if (n == 0) return 1.0;
    if (n > 30) throw std::invalid_argument("permanent: n > 30 is not supported");
    std::vector<cplx_t> row_sum(n, 0.0);
    cplx_t total = 0.0;
    std::uint64_t gray_prev = 0;
    const std::uint64_t count = std::uint64_t(1) << n;
    for (std::uint64_t k = 1; k < count; ++k) {
        const std::uint64_t gray = k ^ (k >> 1);
        const std::uint64_t diff = gray ^ gray_prev;
        int col = 0;
        while (!((diff >> col) & 1)) ++col;
        const double sign = (gray & diff) ? 1.0 : -1.0;   // column added or removed
        for (int r = 0; r < n; ++r) row_sum[r] += sign * a[r * n + col];
        cplx_t prod = 1.0;
        for (int r = 0; r < n; ++r) prod *= row_sum[r];
        int bits = 0;
        for (std::uint64_t g = gray; g; g &= g - 1) ++bits;
        total += ((n - bits) % 2 ? -1.0 : 1.0) * prod;
        gray_prev = gray;
    }
    return total;
}

inline cplx_t hafnian_dp(const std::vector<cplx_t>& a, int n) {
    if (n % 2) return 0.0;
    if (n == 0) return 1.0;
    if (n > 26) throw std::invalid_argument("hafnian: n > 26 is not supported");
    const std::uint32_t full = (std::uint32_t(1) << n) - 1;
    // h[mask] = hafnian of the submatrix on the indices in mask; built from small masks upwards
    std::vector<cplx_t> h(std::size_t(1) << n, 0.0);
    h[0] = 1.0;
    for (std::uint32_t mask = 1; mask <= full; ++mask) {
        int pc = 0;
        for (std::uint32_t m = mask; m; m &= m - 1) ++pc;
        if (pc % 2) continue;
        int i = 0;
        while (!((mask >> i) & 1)) ++i;
        const std::uint32_t rest = mask & ~(std::uint32_t(1) << i);
        cplx_t s = 0.0;
        for (std::uint32_t m = rest; m; m &= m - 1) {
            int j = 0;
            while (!((m >> j) & 1)) ++j;
            s += a[i * n + j] * h[rest & ~(std::uint32_t(1) << j)];
        }
        h[mask] = s;
    }
    return h[full];
}

}  // namespace pqc
