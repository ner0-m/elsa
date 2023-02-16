#include "Quadric.h"
#include "DataContainer.h"
#include "Identity.h"
#include "TypeCasts.hpp"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Quadric<data_t>::Quadric(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : Functional<data_t>(A.getDomainDescriptor()), linResidual_{A, b}
    {
    }

    template <typename data_t>
    Quadric<data_t>::Quadric(const LinearOperator<data_t>& A)
        : Functional<data_t>(A.getDomainDescriptor()), linResidual_{A}
    {
    }

    template <typename data_t>
    Quadric<data_t>::Quadric(const DataContainer<data_t>& b)
        : Functional<data_t>(b.getDataDescriptor()), linResidual_{b}
    {
    }

    template <typename data_t>
    Quadric<data_t>::Quadric(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor), linResidual_{domainDescriptor}
    {
    }

    template <typename data_t>
    const LinearResidual<data_t>& Quadric<data_t>::getGradientExpression() const
    {
        return linResidual_;
    }

    template <typename data_t>
    data_t Quadric<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        data_t xtAx;

        if (linResidual_.hasOperator()) {
            auto temp = linResidual_.getOperator().apply(Rx);
            xtAx = Rx.dot(temp);
        } else {
            xtAx = Rx.squaredL2Norm();
        }

        if (linResidual_.hasDataVector()) {
            return static_cast<data_t>(0.5) * xtAx - Rx.dot(linResidual_.getDataVector());
        } else {
            return static_cast<data_t>(0.5) * xtAx;
        }
    }

    template <typename data_t>
    void Quadric<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                          DataContainer<data_t>& out)
    {
        out = linResidual_.evaluate(Rx);
    }

    template <typename data_t>
    LinearOperator<data_t>
        Quadric<data_t>::getHessianImpl([[maybe_unused]] const DataContainer<data_t>& Rx)
    {
        if (linResidual_.hasOperator())
            return leaf(linResidual_.getOperator());
        else
            return leaf(Identity<data_t>(this->getDomainDescriptor()));
    }

    template <typename data_t>
    Quadric<data_t>* Quadric<data_t>::cloneImpl() const
    {
        if (linResidual_.hasOperator() && linResidual_.hasDataVector())
            return new Quadric<data_t>(linResidual_.getOperator(), linResidual_.getDataVector());
        else if (linResidual_.hasOperator() && !linResidual_.hasDataVector())
            return new Quadric<data_t>(linResidual_.getOperator());
        else if (!linResidual_.hasOperator() && linResidual_.hasDataVector())
            return new Quadric<data_t>(linResidual_.getDataVector());
        else
            return new Quadric<data_t>(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool Quadric<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherQuadric = downcast_safe<Quadric>(&other);
        if (!otherQuadric)
            return false;

        if (linResidual_ != otherQuadric->linResidual_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Quadric<float>;
    template class Quadric<double>;
    template class Quadric<complex<float>>;
    template class Quadric<complex<double>>;

} // namespace elsa
