#pragma once

#include "DataContainer.h"
#include "elsaDefines.h"

namespace elsa
{
    /**
     * The Proximal operator of the Huber functional. The proximal is defined
     * as
     * \[
     * prox_{sigma H}(v) = (1 - (\frac{\sigma}{\max \{ || v ||_2, \sigma\}}))v
     * \]
     *
     * @see Huber
     */
    template <typename data_t = real_t>
    class ProximalHuber
    {
    public:
        ProximalHuber() = default;

        ProximalHuber(data_t delta);

        ~ProximalHuber() = default;

        /**
         * @brief apply the proximal operator of the l1 norm to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         * @param[out] prox output DataContainer
         */
        void apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                   DataContainer<data_t>& prox) const;

        DataContainer<data_t> apply(const DataContainer<data_t>& v, SelfType_t<data_t> t) const;

        data_t delta() const { return delta_; }

    private:
        data_t delta_{0.0001};
    };

    template <typename T>
    bool operator==(const ProximalHuber<T>& lhs, const ProximalHuber<T>& rhs)
    {
        return lhs.delta() == rhs.delta();
    }

    template <typename T>
    bool operator!=(const ProximalHuber<T>& lhs, const ProximalHuber<T>& rhs)
    {
        return !(lhs == rhs);
    }
} // namespace elsa
