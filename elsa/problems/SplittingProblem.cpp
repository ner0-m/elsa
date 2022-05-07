#include "SplittingProblem.h"

namespace elsa
{
    template <typename data_t>
    SplittingProblem<data_t>::SplittingProblem(const Functional<data_t>& f,
                                               const std::vector<RegularizationTerm<data_t>>& g,
                                               const Constraint<data_t>& constraint)
        : Problem<data_t>(f, g), _f{f.clone()}, _g{g}, _constraint{constraint.clone()}
    {
    }

    template <typename data_t>
    SplittingProblem<data_t>::SplittingProblem(const Functional<data_t>& f,
                                               const RegularizationTerm<data_t>& g,
                                               const Constraint<data_t>& constraint)
        : Problem<data_t>(f, g), _f{f.clone()}, _g{g}, _constraint{constraint.clone()}
    {
    }

    template <typename data_t>
    auto SplittingProblem<data_t>::getConstraint() const -> const Constraint<data_t>&
    {
        return *_constraint;
    }

    template <typename data_t>
    auto SplittingProblem<data_t>::cloneImpl() const -> SplittingProblem<data_t>*
    {
        return new SplittingProblem<data_t>(*_f, _g, *_constraint);
    }

    template <typename data_t>
    auto SplittingProblem<data_t>::evaluateImpl() -> data_t
    {
        throw std::runtime_error("SplittingProblem::evaluateImpl: currently unsupported operation");
    }

    template <typename data_t>
    void SplittingProblem<data_t>::getGradientImpl(DataContainer<data_t>&)
    {
        throw std::runtime_error(
            "SplittingProblem::getGradientImpl: currently unsupported operation");
    }

    template <typename data_t>
    auto SplittingProblem<data_t>::getHessianImpl() const -> LinearOperator<data_t>
    {
        throw std::runtime_error(
            "SplittingProblem::getHessianImpl: currently unsupported operation");
    }

    template <typename data_t>
    auto SplittingProblem<data_t>::getLipschitzConstantImpl(index_t) const -> data_t
    {
        throw std::runtime_error(
            "SplittingProblem::getLipschitzConstantImpl: currently unsupported operation");
    }

    template <typename data_t>
    auto SplittingProblem<data_t>::getF() const -> const Functional<data_t>&
    {
        return *_f;
    }

    template <typename data_t>
    auto SplittingProblem<data_t>::getG() const -> const std::vector<RegularizationTerm<data_t>>&
    {
        return _g;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SplittingProblem<float>;
    template class SplittingProblem<complex<float>>;
    template class SplittingProblem<double>;
    template class SplittingProblem<complex<double>>;
} // namespace elsa
