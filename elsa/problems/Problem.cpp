#include "Problem.h"

namespace elsa
{
    template <typename data_t>
    Problem<data_t>::Problem(const DataContainer<data_t>& x0)
        : _currentSolution{x0}
    {}

    template <typename data_t>
    Problem<data_t>::Problem(const DataDescriptor& domainDescriptor)
        : _currentSolution{domainDescriptor}
    {}


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
    data_t Problem<data_t>::evaluate()
    {
        return _evaluate();
    }

    template <typename data_t>
    DataContainer<data_t> Problem<data_t>::getGradient()
    {
        DataContainer<data_t> result(_currentSolution.getDataDescriptor());
        getGradient(result);
        return result;
    }

    template <typename data_t>
    void Problem<data_t>::getGradient(DataContainer<data_t>& result)
    {
        _getGradient(result);
    }

    template <typename data_t>
    LinearOperator<data_t> Problem<data_t>::getHessian()
    {
        return _getHessian();
    }


    template <typename data_t>
    bool Problem<data_t>::isEqual(const Problem<data_t>& other) const
    {
        if (_currentSolution != other._currentSolution)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Problem<float>;
    template class Problem<double>;
    template class Problem<std::complex<float>>;
    template class Problem<std::complex<double>>;

} // namespace elsa
