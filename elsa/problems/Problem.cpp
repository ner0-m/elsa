#include "Problem.h"
#include "Scaling.h"
#include "Logger.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm,
                             const std::vector<RegularizationTerm<data_t>>& regTerms,
                             const std::optional<data_t> lipschitzConstant)
        : _dataTerm{dataTerm.clone()}, _regTerms{regTerms}, _lipschitzConstant{lipschitzConstant}
    {
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm,
                             const RegularizationTerm<data_t>& regTerm,
                             const std::optional<data_t> lipschitzConstant)
        : _dataTerm{dataTerm.clone()}, _regTerms{regTerm}, _lipschitzConstant{lipschitzConstant}
    {
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Functional<data_t>& dataTerm,
                             const std::optional<data_t> lipschitzConstant)
        : _dataTerm{dataTerm.clone()}, _lipschitzConstant{lipschitzConstant}
    {
    }

    template <typename data_t>
    Problem<data_t>::Problem(const Problem<data_t>& problem)
        : Cloneable<Problem<data_t>>(),
          _dataTerm{problem._dataTerm->clone()},
          _regTerms{problem._regTerms},
          _lipschitzConstant{problem._lipschitzConstant}
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
    data_t Problem<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        data_t result = _dataTerm->evaluate(x);

        for (auto& regTerm : _regTerms)
            result += regTerm.getWeight() * regTerm.getFunctional().evaluate(x);

        return result;
    }

    template <typename data_t>
    void Problem<data_t>::getGradientImpl(const DataContainer<data_t>& x,
                                          DataContainer<data_t>& result)
    {
        _dataTerm->getGradient(x, result);

        for (auto& regTerm : _regTerms)
            result += regTerm.getWeight() * regTerm.getFunctional().getGradient(x);
    }

    template <typename data_t>
    LinearOperator<data_t> Problem<data_t>::getHessianImpl(const DataContainer<data_t>& x) const
    {
        auto hessian = _dataTerm->getHessian(x);

        for (auto& regTerm : _regTerms) {
            Scaling weight(x.getDataDescriptor(), regTerm.getWeight());
            hessian = hessian + (weight * regTerm.getFunctional().getHessian(x));
        }

        return hessian;
    }

    template <typename data_t>
    data_t Problem<data_t>::getLipschitzConstantImpl(const DataContainer<data_t>& x,
                                                     index_t nIterations) const
    {
        Timer guard("Problem", "Calculating Lipschitz constant");
        Logger::get("Problem")->info("Calculating Lipschitz constant");

        if (_lipschitzConstant.has_value()) {
            return _lipschitzConstant.value();
        }

        // compute the Lipschitz Constant as the largest eigenvalue of the Hessian
        const auto hessian = getHessian(x);
        DataContainer<data_t> dcB(hessian.getDomainDescriptor());
        dcB = 1;
        for (index_t i = 0; i < nIterations; i++) {
            dcB = hessian.apply(dcB);
            dcB = dcB / dcB.l2Norm();
        }

        return dcB.dot(hessian.apply(dcB)) / dcB.l2Norm();
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
    data_t Problem<data_t>::evaluate(const DataContainer<data_t>& x)
    {
        return evaluateImpl(x);
    }

    template <typename data_t>
    DataContainer<data_t> Problem<data_t>::getGradient(const DataContainer<data_t>& x)
    {
        DataContainer<data_t> result(x.getDataDescriptor(), x.getDataHandlerType());
        getGradient(x, result);
        return result;
    }

    template <typename data_t>
    void Problem<data_t>::getGradient(const DataContainer<data_t>& x, DataContainer<data_t>& result)
    {
        getGradientImpl(x, result);
    }

    template <typename data_t>
    LinearOperator<data_t> Problem<data_t>::getHessian(const DataContainer<data_t>& x) const
    {
        return getHessianImpl(x);
    }

    template <typename data_t>
    data_t Problem<data_t>::getLipschitzConstant(const DataContainer<data_t>& x,
                                                 index_t nIterations) const
    {
        return getLipschitzConstantImpl(x, nIterations);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Problem<float>;

    template class Problem<double>;

    template class Problem<complex<float>>;

    template class Problem<complex<double>>;

} // namespace elsa
