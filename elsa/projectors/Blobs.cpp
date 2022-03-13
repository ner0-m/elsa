#include "Blobs.h"
#include "Bessel.h"

namespace elsa
{
    template <typename data_t>
    constexpr Blob<data_t>::Blob(data_t radius, SelfType_t<data_t> alpha, int order)
        : radius_(radius), alpha_(alpha), order_(order)
    {
    }

    template <typename data_t>
    constexpr data_t Blob<data_t>::operator()(data_t s)
    {
        return blobs::blob_evaluate(s, radius_, alpha_, order_);
    }

    template <typename data_t>
    constexpr data_t Blob<data_t>::radius() const
    {
        return radius_;
    }

    template <typename data_t>
    constexpr data_t Blob<data_t>::alpha() const
    {
        return alpha_;
    }

    template <typename data_t>
    constexpr int Blob<data_t>::order() const
    {
        return order_;
    }
    // ------------------------------------------
    // explicit template instantiation
    namespace blobs
    {
        template float blob_evaluate<float>(float, SelfType_t<float>, SelfType_t<float>, int);
        template double blob_evaluate<double>(double, SelfType_t<double>, SelfType_t<double>, int);

        template float blob_projected<float>(float, SelfType_t<float>, SelfType_t<float>, int);
        template double blob_projected<double>(double, SelfType_t<double>, SelfType_t<double>, int);

        template float blob_projected<float>(float);
        template double blob_projected<double>(double);
    } // namespace blobs

    template class Blob<float>;
    template class Blob<double>;
} // namespace elsa
