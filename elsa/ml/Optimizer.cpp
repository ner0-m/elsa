#include "Optimizer.h"

#include <limits>
#include <iostream>
namespace elsa
{

    template <typename data_t>
    AdamOptimizer<data_t>::AdamOptimizer(index_t size, data_t learningRate, data_t beta1,
                                         data_t beta2)
        : _size(size),
          _learningRate(learningRate),
          _beta1(beta1),
          _beta2(beta2),
          _m(size, static_cast<data_t>(0)),
          _v(size, static_cast<data_t>(0))
    {
    }

    template <typename data_t>
    void AdamOptimizer<data_t>::operator++()
    {
        _iteration++;
    }

    template <typename data_t>
    data_t AdamOptimizer<data_t>::getUpdateValue(const data_t& gradient, std::size_t index)
    {
        _m[index] = _beta1 * _m[index] + (1 - _beta1) * gradient;
        _v[index] = _beta2 * _v[index] + (1 - _beta2) * gradient * gradient;

        data_t mHat = _m[index] / (1 - std::pow(_beta1, _iteration));
        data_t vHat = _v[index] / (1 - std::pow(_beta2, _iteration));

        return _learningRate * mHat
               / static_cast<data_t>(std::sqrt(vHat) + static_cast<data_t>(1e-8));
    }

    template class AdamOptimizer<float>;
} // namespace elsa
