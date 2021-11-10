#pragma once

#include <cstddef>
#include <Eigen/Core>
#include <type_traits>
#include <thrust/complex.h>

namespace quickvec
{
    using real_t = float;                      ///< global type for real numbers
    using complex_t = thrust::complex<real_t>; ///< global type for complex numbers
    using index_t = std::ptrdiff_t;            ///< global type for indices
} // namespace quickvec
