#include "RegularizationTerm.h"
#include "L1Norm.h"

namespace elsa
{
    template <typename data_t>
    RegularizationTerm<data_t>::RegularizationTerm(data_t weight,
                                                   const Functional<data_t>& functional)
        : _weight{weight}, _functional{functional.clone()}
    {
    }

    template <typename data_t>
    RegularizationTerm<data_t>::RegularizationTerm(const RegularizationTerm<data_t>& other)
        : _weight{other._weight}, _functional{other._functional->clone()}
    {
    }

    template <typename data_t>
    RegularizationTerm<data_t>& RegularizationTerm<data_t>::
        operator=(const RegularizationTerm<data_t>& other)
    {
        if (this != &other) {
            _weight = other._weight;
            _functional = other._functional->clone();
        }

        return *this;
    }

    template <typename data_t>
    RegularizationTerm<data_t>::RegularizationTerm(RegularizationTerm<data_t>&& other)
        : _weight{std::move(other._weight)}, _functional{std::move(other._functional)}
    {
        // make sure we leave other in a valid state (since we do not check for empty pointers!)
        other._functional = std::make_unique<L1Norm<data_t>>(_functional->getDomainDescriptor());
    }

    template <typename data_t>
    RegularizationTerm<data_t>& RegularizationTerm<data_t>::
        operator=(RegularizationTerm<data_t>&& other)
    {
        _weight = std::move(other._weight);
        _functional = std::move(other._functional);

        // make sure we leave other in a valid state (since we do not check for empty pointers!)
        other._functional = std::make_unique<L1Norm<data_t>>(_functional->getDomainDescriptor());

        return *this;
    }

    template <typename data_t>
    data_t RegularizationTerm<data_t>::getWeight() const
    {
        return _weight;
    }

    template <typename data_t>
    Functional<data_t>& RegularizationTerm<data_t>::getFunctional() const
    {
        return *_functional;
    }

    template <typename data_t>
    bool RegularizationTerm<data_t>::operator==(const RegularizationTerm<data_t>& other) const
    {
        return (_weight == other._weight && *_functional == *other._functional);
    }

    template <typename data_t>
    bool RegularizationTerm<data_t>::operator!=(const RegularizationTerm<data_t>& other) const
    {
        return !operator==(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class RegularizationTerm<float>;
    template class RegularizationTerm<double>;
    template class RegularizationTerm<std::complex<float>>;
    template class RegularizationTerm<std::complex<double>>;

} // namespace elsa