#pragma once

#include <optional>

#include "DataContainer.h"
#include "StrongTypes.h"

namespace elsa
{
    /**
     * @brief Proximal operator for the functional \f$ || x - b ||_2^2\f$, where
     * \f$b\f$ is potentially zero.
     */
    template <class data_t>
    class ProximalL2Squared
    {
    public:
        ProximalL2Squared() = default;

        ProximalL2Squared(const DataContainer<data_t>& b);

        DataContainer<data_t> apply(const DataContainer<data_t>& v,
                                    geometry::Threshold<data_t> t) const;

        void apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                   DataContainer<data_t>& prox) const;

    private:
        std::optional<DataContainer<data_t>> b_ = {};
    };
} // namespace elsa