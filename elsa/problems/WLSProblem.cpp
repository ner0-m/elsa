#include "WLSProblem.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"

namespace elsa
{
    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const Scaling<data_t>& W, const LinearOperator<data_t>& A,
                                   const DataContainer<data_t>& b, const DataContainer<data_t>& x0)
        : Problem<data_t>{WeightedL2NormPow2<data_t>{LinearResidual<data_t>{A, b}, W}, x0}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const Scaling<data_t>& W, const LinearOperator<data_t>& A,
                                   const DataContainer<data_t>& b)
        : Problem<data_t>{WeightedL2NormPow2<data_t>{LinearResidual<data_t>{A, b}, W}}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                                   const DataContainer<data_t>& x0)
        : Problem<data_t>{L2NormPow2<data_t>{LinearResidual<data_t>{A, b}}, x0}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : Problem<data_t>{L2NormPow2<data_t>{LinearResidual<data_t>{A, b}}}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>* WLSProblem<data_t>::cloneImpl() const
    {
        return new WLSProblem(*this);
    }

    template <typename data_t>
    bool WLSProblem<data_t>::isEqual(const Problem<data_t>& other) const
    {
        if (!Problem<data_t>::isEqual(other))
            return false;

        auto otherWLS = dynamic_cast<const WLSProblem*>(&other);
        if (!otherWLS)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class WLSProblem<float>;
    template class WLSProblem<double>;
    template class WLSProblem<std::complex<float>>;
    template class WLSProblem<std::complex<double>>;

} // namespace elsa
