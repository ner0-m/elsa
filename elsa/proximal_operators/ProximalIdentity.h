#pragma once

#include "DataContainer.h"
#include "elsaDefines.h"

namespace elsa
{
    /**
     * @brief Proximal operator which maps the vector to itself. This is the
     * proximal operator for all constant functionals.
     */
    template <class data_t>
    class ProximalIdentity
    {
    public:
        ProximalIdentity() = default;

        DataContainer<data_t> apply(const DataContainer<data_t>& v, SelfType_t<data_t>) const
        {
            return v;
        }

        void apply(const DataContainer<data_t>& v, SelfType_t<data_t>,
                   DataContainer<data_t>& prox) const
        {
            prox = v;
        }
    };
} // namespace elsa
