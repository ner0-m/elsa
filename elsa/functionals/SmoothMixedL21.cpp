#include "SmoothMixedL21.h"
#include "BlockDescriptor.h"
#include "BlockLinearOperator.h"
#include "Identity.h"
#include "Scaling.h"
#include "functions/Square.hpp"

using namespace std;

namespace elsa
{
    template <typename data_t>
    SmoothMixedL21<data_t>::SmoothMixedL21(const DataDescriptor& domainDescriptor, data_t epsilon)
        : Functional<data_t>(domainDescriptor), epsilon{epsilon}
    {
    }

    template <typename data_t>
    data_t SmoothMixedL21<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {;
        return Rx.l21SmoothMixedNorm(epsilon);
    }

    template <typename data_t>
    void SmoothMixedL21<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                                DataContainer<data_t>& out)
    {

        auto tmp = DataContainer<data_t>(Rx.getBlock(0).getDataDescriptor());

        for (index_t i = 0; i < Rx.getNumberOfBlocks(); ++i) {
            tmp += (square(Rx.getBlock(i)));
        }

        tmp += (epsilon * epsilon);
        tmp = sqrt(tmp);

        for (index_t i = 0; i < Rx.getNumberOfBlocks(); ++i) {
            out.getBlock(i) = Rx.getBlock(i) / tmp;
        }

    }
    template <typename data_t>
    LinearOperator<data_t> SmoothMixedL21<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {

        auto tmp = DataContainer<data_t>(Rx.getBlock(0).getDataDescriptor());

        for (index_t i = 0; i < Rx.getNumberOfBlocks(); ++i) {
            tmp += (square(Rx.getBlock(i)));
        }

        tmp += (epsilon * epsilon);
        tmp = tmp*sqrt(tmp);
        tmp = (epsilon * epsilon) / tmp;

        DataContainer<data_t> scale_factors(Rx.getDataDescriptor());

        for (index_t i = 0; i < Rx.getNumberOfBlocks(); ++i) {
            scale_factors.getBlock(i) = tmp * Rx.getBlock(i);
        }

        return leaf(Scaling<data_t>(Rx.getDataDescriptor(), scale_factors));

    }
    template <typename data_t>
    SmoothMixedL21<data_t>* SmoothMixedL21<data_t>::cloneImpl() const
    {
        return new SmoothMixedL21(this->getDomainDescriptor(), epsilon);
    }
    template <typename data_t>
    bool SmoothMixedL21<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        return is<SmoothMixedL21>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SmoothMixedL21<float>;
    template class SmoothMixedL21<double>;
    template class SmoothMixedL21<complex<float>>;
    template class SmoothMixedL21<complex<double>>;

} // namespace elsa