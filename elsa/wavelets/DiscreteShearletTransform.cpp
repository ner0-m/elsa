#include "DiscreteShearletTransform.h"

namespace elsa
{
    // TODO the inputs here should be enough to define the entire system
    template <typename data_t>
    DiscreteShearletTransform<data_t>::DiscreteShearletTransform(int j, int k, std::vector<int> m)
        : LinearOperator<data_t>(, , )
    {
        if (m.size() != 2) {
            throw InvalidArgumentError(
                "DiscreteShearletTransform: the position index m is two-dimensional");
        }
    }

    template <typename data_t>
    void DiscreteShearletTransform<data_t>::applyImpl(const DataContainer<data_t>& f,
                                                      DataContainer<data_t>& SHf) const
    {
    }

    template <typename data_t>
    void DiscreteShearletTransform<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                             DataContainer<data_t>& SHty) const
    {
    }

    template <typename data_t>
    DiscreteShearletTransform<data_t>* DiscreteShearletTransform<data_t>::cloneImpl() const
    {
    }

    template <typename data_t>
    bool DiscreteShearletTransform<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
    }

    template <typename data_t>
    DataContainer<data_t> DiscreteShearletTransform<data_t>::psi(int j, int k, std::vector<int> m)
    {
    }

    template <typename data_t>
    DataContainer<data_t> DiscreteShearletTransform<data_t>::phi(int j, int k, std::vector<int> m)
    {
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DiscreteShearletTransform<float>;
    template class DiscreteShearletTransform<double>;
    // TODO what about complex types
} // namespace elsa
