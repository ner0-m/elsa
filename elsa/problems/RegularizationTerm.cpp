#include "RegularizationTerm.h"
#include "L1Norm.h"
#include "elsaDefines.h"

namespace elsa
{
    template <typename data_t>
    RegularizationTerm<data_t>::RegularizationTerm(GetFloatingPointType_t<data_t> weight,
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
    RegularizationTerm<data_t>&
        RegularizationTerm<data_t>::operator=(const RegularizationTerm<data_t>& other)
    {
        if (this != &other) {
            _weight = other._weight;
            _functional = other._functional->clone();
        }

        return *this;
    }

    template <typename data_t>
    RegularizationTerm<data_t>::RegularizationTerm(RegularizationTerm<data_t>&& other) noexcept
        : _weight{std::move(other._weight)}, _functional{std::move(other._functional)}
    {
    }

    template <typename data_t>
    RegularizationTerm<data_t>&
        RegularizationTerm<data_t>::operator=(RegularizationTerm<data_t>&& other) noexcept
    {
        _weight = std::move(other._weight);
        _functional = std::move(other._functional);

        return *this;
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> RegularizationTerm<data_t>::getWeight() const
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
