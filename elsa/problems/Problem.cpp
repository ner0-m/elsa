#include "Problem.h"
#include "Scaling.h"

namespace elsa
{
    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm,
                             const std::vector<RegularizationTerm<data_t>>& regTerms,
                             const DataContainer<data_t>& x0)
        : _dataTerm{dataTerm.clone()}, _regTerms{regTerms}, _currentSolution{x0}
    {
        // sanity checks
        if (_dataTerm->getDomainDescriptor().getNumberOfCoefficients()
            != this->_currentSolution.getSize())
            throw std::invalid_argument("Problem: domain of dataTerm and solution do not match");

        for (auto& regTerm : _regTerms) {
            if (dataTerm.getDomainDescriptor().getNumberOfCoefficients()
                != regTerm.getFunctional().getDomainDescriptor().getNumberOfCoefficients())
                throw std::invalid_argument(
                    "Problem: one of the reg terms' domain does not match the data term's");
        }
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm,
                             const std::vector<RegularizationTerm<data_t>>& regTerms)
        : _dataTerm{dataTerm.clone()},
          _regTerms{regTerms},
          _currentSolution{dataTerm.getDomainDescriptor()}
    {
        // sanity check
        for (auto& regTerm : _regTerms) {
            if (dataTerm.getDomainDescriptor().getNumberOfCoefficients()
                != regTerm.getFunctional().getDomainDescriptor().getNumberOfCoefficients())
                throw std::invalid_argument(
                    "Problem: one of the reg terms' domain does not match the data term's");
        }

        _currentSolution = 0;
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm,
                             const RegularizationTerm<data_t>& regTerm,
                             const DataContainer<data_t>& x0)
        : _dataTerm{dataTerm.clone()}, _regTerms{regTerm}, _currentSolution{x0}
    {
        // sanity checks
        if (_dataTerm->getDomainDescriptor().getNumberOfCoefficients()
            != this->_currentSolution.getSize())
            throw std::invalid_argument("Problem: domain of dataTerm and solution do not match");

        if (dataTerm.getDomainDescriptor().getNumberOfCoefficients()
            != regTerm.getFunctional().getDomainDescriptor().getNumberOfCoefficients())
            throw std::invalid_argument(
                "Problem: one of the reg terms' domain does not match the data term's");
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm,
                             const RegularizationTerm<data_t>& regTerm)
        : _dataTerm{dataTerm.clone()},
          _regTerms{regTerm},
          _currentSolution{dataTerm.getDomainDescriptor(), defaultHandlerType}
    {
        // sanity check
        if (dataTerm.getDomainDescriptor().getNumberOfCoefficients()
            != regTerm.getFunctional().getDomainDescriptor().getNumberOfCoefficients())
            throw std::invalid_argument(
                "Problem: one of the reg terms' domain does not match the data term's");

        _currentSolution = 0;
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm, const DataContainer<data_t>& x0)
        : _dataTerm{dataTerm.clone()}, _currentSolution{x0}
    {
        // sanity check
        if (_dataTerm->getDomainDescriptor().getNumberOfCoefficients()
            != this->_currentSolution.getSize())
            throw std::invalid_argument("Problem: domain of dataTerm and solution do not match");
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm)
        : _dataTerm{dataTerm.clone()},
          _currentSolution{dataTerm.getDomainDescriptor(), defaultHandlerType}
    {
        _currentSolution = 0;
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Problem<data_t>& problem)
        : Cloneable<Problem<data_t>>(),
          _dataTerm{problem._dataTerm->clone()},
          _regTerms{problem._regTerms},
          _currentSolution{problem._currentSolution}
    {
    }

    template <typename data_t>
    const Functional<data_t>& Problem<data_t>::getDataTerm() const
    {
        return *_dataTerm;
    }

    template <typename data_t>
    const std::vector<RegularizationTerm<data_t>>& Problem<data_t>::getRegularizationTerms() const
    {
        return _regTerms;
    }

    template <typename data_t>
    const DataContainer<data_t>& Problem<data_t>::getCurrentSolution() const
    {
        return _currentSolution;
    }

    template <typename data_t>
    DataContainer<data_t>& Problem<data_t>::getCurrentSolution()
    {
        return _currentSolution;
    }

    template <typename data_t>
    data_t Problem<data_t>::evaluateImpl()
    {
        data_t result = _dataTerm->evaluate(_currentSolution);

        for (auto& regTerm : _regTerms)
            result += regTerm.getWeight() * regTerm.getFunctional().evaluate(_currentSolution);

        return result;
    }

    template <typename data_t>
    void Problem<data_t>::getGradientImpl(DataContainer<data_t>& result)
    {
        _dataTerm->getGradient(_currentSolution, result);

        for (auto& regTerm : _regTerms)
            result += regTerm.getWeight() * regTerm.getFunctional().getGradient(_currentSolution);
    }

    template <typename data_t>
    LinearOperator<data_t> Problem<data_t>::getHessianImpl() const
    {
        auto hessian = _dataTerm->getHessian(_currentSolution);

        for (auto& regTerm : _regTerms) {
            Scaling weight(_currentSolution.getDataDescriptor(), regTerm.getWeight());
            hessian = hessian + (weight * regTerm.getFunctional().getHessian(_currentSolution));
        }

        return hessian;
    }

    template <typename data_t>
    data_t Problem<data_t>::getLipschitzConstantImpl(index_t nIterations) const
    {
        // compute the Lipschitz Constant as the largest eigenvalue of the Hessian
        const auto hessian = getHessian();
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> bVec(
            hessian.getDomainDescriptor().getNumberOfCoefficients());
        bVec.setOnes();
        DataContainer<data_t> dcB(hessian.getDomainDescriptor(), bVec);
        for (index_t i = 0; i < nIterations; i++) {
            dcB = hessian.apply(dcB);
            dcB = dcB / std::sqrt(dcB.dot(dcB));
        }

        return dcB.dot(hessian.apply(dcB)) / (dcB.dot(dcB));
    }

    template <typename data_t>
    Problem<data_t>* Problem<data_t>::cloneImpl() const
    {
        return new Problem(*this);
    }

    template <typename data_t>
    bool Problem<data_t>::isEqual(const Problem<data_t>& other) const
    {
        if (typeid(*this) != typeid(other))
            return false;

        if (_currentSolution != other._currentSolution)
            return false;

        if (*_dataTerm != *other._dataTerm)
            return false;

        if (_regTerms.size() != other._regTerms.size())
            return false;

        for (std::size_t i = 0; i < _regTerms.size(); ++i)
            if (_regTerms.at(i) != other._regTerms.at(i))
                return false;

        return true;
    }

    template <typename data_t>
    data_t Problem<data_t>::evaluate()
    {
        return evaluateImpl();
    }

    template <typename data_t>
    DataContainer<data_t> Problem<data_t>::getGradient()
    {
        DataContainer<data_t> result(_currentSolution.getDataDescriptor(),
                                     _currentSolution.getDataHandlerType());
        getGradient(result);
        return result;
    }

    template <typename data_t>
    void Problem<data_t>::getGradient(DataContainer<data_t>& result)
    {
        getGradientImpl(result);
    }

    template <typename data_t>
    LinearOperator<data_t> Problem<data_t>::getHessian() const
    {
        return getHessianImpl();
    }

    template <typename data_t>
    data_t Problem<data_t>::getLipschitzConstant(index_t nIterations) const
    {
        return getLipschitzConstantImpl(nIterations);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Problem<float>;

    template class Problem<double>;

    template class Problem<std::complex<float>>;

    template class Problem<std::complex<double>>;

} // namespace elsa
