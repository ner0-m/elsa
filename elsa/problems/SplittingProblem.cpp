#include "SplittingProblem.h"
#include "Identity.h"
#include "RegularizationTerm.h"
#include "Scaling.h"

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
    SplittingProblem<data_t>::SplittingProblem(const Functional<data_t>& f,
                                               const RegularizationTerm<data_t>& g,
                                               const DataDescriptor& rangeDescriptorOfConstraint)
        : Problem<data_t>(f, g), _f{f.clone()}, _g{g}
    {
        const DataDescriptor& dd = g.getFunctional().getDomainDescriptor();
        if (dd != g.getFunctional().getDomainDescriptor()) {
            throw InvalidArgumentError("SplittingProblem: reg. term has different domain "
                                       "descriptor of functional, use another constructor to "
                                       "explicitly build the constraint instead");
        }

        Identity<data_t> A(f.getDomainDescriptor(), rangeDescriptorOfConstraint);
        Scaling<data_t> B(g.getFunctional().getDomainDescriptor(), rangeDescriptorOfConstraint, -1);
        DataContainer<data_t> c(rangeDescriptorOfConstraint);
        c = 0;
        Constraint<data_t> constraint(A, B, c);
        _constraint = constraint.clone();
    }

    template <typename data_t>
    SplittingProblem<data_t>::SplittingProblem(const Functional<data_t>& f,
                                               const RegularizationTerm<data_t>& g)
        : SplittingProblem<data_t>(f, std::vector{g})
    {
    }

    template <typename data_t>
    SplittingProblem<data_t>::SplittingProblem(const Functional<data_t>& f,
                                               const std::vector<RegularizationTerm<data_t>>& g)
        : Problem<data_t>(f, g), _f{f.clone()}, _g{g}
    {
        const DataDescriptor& dd = g[0].getFunctional().getDomainDescriptor();
        for (unsigned long i = 0; i < g.size(); ++i) {
            if (dd != g[i].getFunctional().getDomainDescriptor()) {
                throw InvalidArgumentError("SplittingProblem: reg. terms have different domain "
                                           "descriptors of functionals, use another constructor to "
                                           "explicitly build the constraint instead");
            }
        }

        if (dd != f.getDomainDescriptor()) {
            throw InvalidArgumentError("SplittingProblem: different domain descriptor of the reg "
                                       "terms and the functional, use another constructor to "
                                       "explicitly build the constraint instead");
        }

        Identity<data_t> A(f.getDomainDescriptor());
        Scaling<data_t> B(f.getDomainDescriptor(), -1);
        DataContainer<data_t> c(f.getDomainDescriptor());
        c = 0;
        Constraint<data_t> constraint(A, B, c);
        _constraint = constraint.clone();
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
