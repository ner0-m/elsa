#pragma once

#include <vector>

#include "elsaDefines.h"

namespace elsa
{
    template <typename data_t>
    class AdamOptimizer
    {
    public:
        AdamOptimizer() = default;

        AdamOptimizer(index_t size, data_t learningRate = static_cast<data_t>(0.2e-8f),
                      data_t beta1 = static_cast<data_t>(0.9),
                      data_t beta2 = static_cast<data_t>(0.999));

        data_t getUpdateValue(const data_t& gradient, std::size_t index);

        void operator++();

    private:
        index_t _size;
        data_t _learningRate;
        data_t _beta1;
        data_t _beta2;
        index_t _iteration = 0;
        std::vector<data_t> _m;
        std::vector<data_t> _v;
    };
} // namespace elsa