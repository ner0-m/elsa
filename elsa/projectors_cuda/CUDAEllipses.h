#pragma once

#include <cuda_runtime.h>
#include <Eigen/Core>

#include "Cloneable.h"

#include "DataDescriptor.h"
#include "DetectorDescriptor.h"
#include "TypeCasts.hpp"

#include "analytical/Ellipse.h"
#include "elsaDefines.h"
#include "DataContainer.h"

#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace elsa::phantoms
{

    template <typename data_t, int n>
    class CUDAEllipses
    {
    public:
        using Ray = Eigen::ParametrizedLine<data_t, n>;

        DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor);

        CUDAEllipses& operator+=(const Ellipsoid<n, data_t>& ellipse)
        {
            components.push_back(Functor{ellipse.w, ellipse.c, ellipse.A, ellipse.R});
            return *this;
        }

        CUDAEllipses& operator-=(const Ellipsoid<n, data_t>& ellipse)
        {
            components.push_back(Functor{-ellipse.w, ellipse.c, ellipse.A, ellipse.R});
            return *this;
        }

        struct Functor : thrust::binary_function<data_t, Ray, data_t> {
            Functor(data_t w, Eigen::Vector<data_t, n> c, Eigen::DiagonalMatrix<data_t, n> A,
                    Eigen::Matrix<data_t, n, n> R)
                : w(w), c(c), A(A), R(R)
            {
            }
            const data_t w;
            const Eigen::Vector<data_t, n> c;
            const Eigen::Matrix<data_t, n, n> A; // Eigen::DiagonalMatrix cannot be use in CUDA...
            const Eigen::Matrix<data_t, n, n> R;
            data_t __host__ __device__ operator()(const data_t& I, const Ray& ray) const;
        };
        std::vector<Functor> components;
    };

    template class CUDAEllipses<float, 2>;
    template class CUDAEllipses<float, 3>;
    template class CUDAEllipses<double, 2>;
    template class CUDAEllipses<double, 3>;

} // namespace elsa::phantoms
