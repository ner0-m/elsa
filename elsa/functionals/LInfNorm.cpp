#include "LInfNorm.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    LInfNorm<data_t>::LInfNorm(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {}

    template <typename data_t>
    LInfNorm<data_t>::LInfNorm(const Residual<data_t>& residual)
        : Functional<data_t>(residual)
    {}

    template <typename data_t>
    data_t LInfNorm<data_t>::_evaluate(const DataContainer<data_t>& Rx)
    {
        return Rx.lInfNorm();
    }

    template <typename data_t>
    void LInfNorm<data_t>::_getGradientInPlace(DataContainer<data_t>& Rx)
    {
        throw std::logic_error("LInfNorm: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    LinearOperator<data_t> LInfNorm<data_t>::_getHessian(const DataContainer<data_t>& Rx)
    {
        throw std::logic_error("LInfNorm: not differentiable, so no Hessian! (busted!)");
    }


    template <typename data_t>
    LInfNorm<data_t>* LInfNorm<data_t>::cloneImpl() const
    {
        return new LInfNorm(this->getResidual());
    }

    template <typename data_t>
    bool LInfNorm<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherLInfNorm = dynamic_cast<const LInfNorm*>(&other);
        if (!otherLInfNorm)
            return false;

        return true;
    }


    // ------------------------------------------
    // explicit template instantiation
    template class LInfNorm<float>;
    template class LInfNorm<double>;
    template class LInfNorm<std::complex<float>>;
    template class LInfNorm<std::complex<double>>;

} // namespace elsa
