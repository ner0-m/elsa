#include "WLSProblem.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"

namespace elsa
{
    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const Scaling<data_t>& W, const LinearOperator<data_t>& A,
                                   const DataContainer<data_t>& b, const DataContainer<data_t>& x0)
        : Problem<data_t>(x0), _residual{std::make_unique<LinearResidual<data_t>>(A, b)},
          _functional{std::make_unique<WeightedL2NormPow2<data_t>>(*_residual, W)}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const Scaling<data_t>& W, const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : Problem<data_t>(A.getDomainDescriptor()), _residual{std::make_unique<LinearResidual<data_t>>(A, b)},
          _functional{std::make_unique<WeightedL2NormPow2<data_t>>(*_residual, W)}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b, const DataContainer<data_t>& x0)
        : Problem<data_t>(x0), _residual{std::make_unique<LinearResidual<data_t>>(A, b)},
          _functional{std::make_unique<L2NormPow2<data_t>>(*_residual)}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : Problem<data_t>(A.getDomainDescriptor()), _residual{std::make_unique<LinearResidual<data_t>>(A, b)},
          _functional{std::make_unique<L2NormPow2<data_t>>(*_residual)}
    {
        // sanity checks are done in the member constructors already
    }


    template <typename data_t>
    data_t WLSProblem<data_t>::_evaluate()
    {
        return _functional->evaluate(this->getCurrentSolution());
    }

    template <typename data_t>
    void WLSProblem<data_t>::_getGradient(DataContainer<data_t>& result)
    {
        _functional->getGradient(this->getCurrentSolution(), result);
    }

    template <typename data_t>
    LinearOperator<data_t> WLSProblem<data_t>::_getHessian()
    {
        return _functional->getHessian(this->getCurrentSolution());
    }


    template <typename data_t>
    WLSProblem<data_t>* WLSProblem<data_t>::cloneImpl() const
    {
        return new WLSProblem(*_residual, *_functional, this->getCurrentSolution());
    }

    template <typename data_t>
    bool WLSProblem<data_t>::isEqual(const Problem<data_t>& other) const
    {
        if (!Problem<data_t>::isEqual(other))
            return false;

        auto otherWLS = dynamic_cast<const WLSProblem*>(&other);
        if (!otherWLS)
            return false;

        if (*_residual != *otherWLS->_residual || *_functional != *otherWLS->_functional)
            return false;

        return true;
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const Residual<data_t>& residual, const Functional<data_t>& functional,
                                   const DataContainer<data_t>& x0)
        : Problem<data_t>(x0), _residual{residual.clone()}, _functional{functional.clone()}
    {}


    // ------------------------------------------
    // explicit template instantiation
    template class WLSProblem<float>;
    template class WLSProblem<double>;
    template class WLSProblem<std::complex<float>>;
    template class WLSProblem<std::complex<double>>;

} // namespace elsa
