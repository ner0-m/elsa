#include "OptimizationProblem.h"
#include "Scaling.h"

namespace elsa
{
    template <typename data_t>
    OptimizationProblem<data_t>::OptimizationProblem(const Functional<data_t>& dataTerm,
                                                     const std::vector<RegularizationTerm<data_t>>& regTerms,
                                                     const DataContainer<data_t>& x0)
    : Problem<data_t>(x0), _dataTerm{dataTerm.clone()}, _regTerms{regTerms}
    {
        // sanity checks
        if (_dataTerm->getDomainDescriptor() != this->_currentSolution.getDataDescriptor())
            throw std::invalid_argument("OptimizationProblem: domain of dataTerm and solution do not match");

        for (auto& regTerm : _regTerms) {
            if (dataTerm.getDomainDescriptor() != regTerm.getFunctional().getDomainDescriptor())
                throw std::invalid_argument("OptimizationProblem: one of the reg terms' domain does not match the data term's");
        }
    }

    template <typename data_t>
    OptimizationProblem<data_t>::OptimizationProblem(const Functional<data_t>& dataTerm,
                                                     const std::vector<RegularizationTerm<data_t>>& regTerms)
    : Problem<data_t>(dataTerm.getDomainDescriptor()), _dataTerm{dataTerm.clone()}, _regTerms{regTerms}
    {
        // sanity check
        for (auto& regTerm : _regTerms) {
            if (dataTerm.getDomainDescriptor() != regTerm.getFunctional().getDomainDescriptor())
                throw std::invalid_argument("OptimizationProblem: one of the reg terms' domain does not match the data term's");
        }
    }

    template <typename data_t>
    OptimizationProblem<data_t>::OptimizationProblem(const Functional<data_t>& dataTerm, const RegularizationTerm<data_t>& regTerm,
                                                     const DataContainer<data_t>& x0)
    : Problem<data_t>(x0), _dataTerm{dataTerm.clone()}, _regTerms{regTerm}
    {
        // sanity checks
        if (_dataTerm->getDomainDescriptor() != this->_currentSolution.getDataDescriptor())
            throw std::invalid_argument("OptimizationProblem: domain of dataTerm and solution do not match");

        if (dataTerm.getDomainDescriptor() != regTerm.getFunctional().getDomainDescriptor())
            throw std::invalid_argument("OptimizationProblem: one of the reg terms' domain does not match the data term's");
    }

    template <typename data_t>
    OptimizationProblem<data_t>::OptimizationProblem(const Functional<data_t>& dataTerm,
                                                     const RegularizationTerm<data_t>& regTerm)
    : Problem<data_t>(dataTerm.getDomainDescriptor()), _dataTerm{dataTerm.clone()}, _regTerms{regTerm}
    {
        // sanity check
        if (dataTerm.getDomainDescriptor() != regTerm.getFunctional().getDomainDescriptor())
            throw std::invalid_argument("OptimizationProblem: one of the reg terms' domain does not match the data term's");
    }

    template <typename data_t>
    OptimizationProblem<data_t>::OptimizationProblem(const Functional<data_t>& dataTerm, const DataContainer<data_t>& x0)
    : Problem<data_t>(x0), _dataTerm{dataTerm.clone()}
    {
        // sanity check
        if (_dataTerm->getDomainDescriptor() != this->_currentSolution.getDataDescriptor())
            throw std::invalid_argument("OptimizationProblem: domain of dataTerm and solution do not match");
    }

    template <typename data_t>
    OptimizationProblem<data_t>::OptimizationProblem(const Functional<data_t>& dataTerm)
    : Problem<data_t>(dataTerm.getDomainDescriptor()), _dataTerm{dataTerm.clone()}
    {}


    template <typename data_t>
    data_t OptimizationProblem<data_t>::_evaluate()
    {
        data_t result = _dataTerm->evaluate(this->getCurrentSolution());

        for (auto& regTerm : _regTerms)
            result += regTerm.getWeight() * regTerm.getFunctional().evaluate(this->getCurrentSolution());

        return result;
    }

    template <typename data_t>
    void OptimizationProblem<data_t>::_getGradient(DataContainer<data_t>& result)
    {
        _dataTerm->getGradient(this->getCurrentSolution(), result);

        for (auto& regTerm : _regTerms)
            result += regTerm.getWeight() * regTerm.getFunctional().getGradient(this->getCurrentSolution());
    }

    template <typename data_t>
    LinearOperator<data_t> OptimizationProblem<data_t>::_getHessian()
    {
        auto hessian = _dataTerm->getHessian(this->getCurrentSolution());

        for (auto& regTerm : _regTerms) {
            Scaling weight(this->getCurrentSolution().getDataDescriptor(), regTerm.getWeight());
            hessian = hessian + (weight * regTerm.getFunctional().getHessian(this->getCurrentSolution()));
        }

        return hessian;
    }


    template <typename data_t>
    OptimizationProblem<data_t>* OptimizationProblem<data_t>::cloneImpl() const
    {
        return new OptimizationProblem(*_dataTerm, _regTerms, this->getCurrentSolution());
    }

    template <typename data_t>
    bool OptimizationProblem<data_t>::isEqual(const Problem<data_t>& other) const
    {
        if (!Problem<data_t>::isEqual(other))
            return false;

        auto otherOP = dynamic_cast<const OptimizationProblem*>(&other);
        if (!otherOP)
            return false;

        if (*_dataTerm != *otherOP->_dataTerm)
            return false;

        for (index_t i = 0; i < _regTerms.size(); ++i)
            if (_regTerms.at(i) != otherOP->_regTerms.at(i))
                return false;

        return true;
    }


    // ------------------------------------------
    // explicit template instantiation
    template class OptimizationProblem<float>;
    template class OptimizationProblem<double>;
    template class OptimizationProblem<std::complex<float>>;
    template class OptimizationProblem<std::complex<double>>;

} // namespace elsa
