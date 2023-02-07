#pragma once

#include <optional>

#include "DataContainer.h"
#include "StrongTypes.h"
#include "elsaDefines.h"

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

        explicit ProximalL2Squared(data_t sigma);

        explicit ProximalL2Squared(const DataContainer<data_t>& b);

        ProximalL2Squared(const DataContainer<data_t>& b, SelfType_t<data_t> sigma);

        DataContainer<data_t> apply(const DataContainer<data_t>& v, SelfType_t<data_t> t) const;

        void apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                   DataContainer<data_t>& prox) const;

    private:
        data_t sigma_{1};

        std::optional<DataContainer<data_t>> b_ = {};
    };
} // namespace elsa
