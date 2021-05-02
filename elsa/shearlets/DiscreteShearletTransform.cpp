#include <VolumeDescriptor.h>
#include "DiscreteShearletTransform.h"

namespace elsa
{
    // TODO the inputs here should be enough to define the entire system
    template <typename data_t>
    DiscreteShearletTransform<data_t>::DiscreteShearletTransform(index_t width, index_t height,
                                                                 index_t numberOfScales)
        // dummy values for the LinearOperator constructor
        : LinearOperator<data_t>(VolumeDescriptor{{1, 1}}, VolumeDescriptor{{1, 1}}),
          _width{width},
          _height{height},
          _numberOfScales{numberOfScales}
    {
        // sanity check the parameters here
        //        if (scales something) {
        //            throw InvalidArgumentError(
        //                "DiscreteShearletTransform: the allowed number of scales is ... ");
        //        }

        // TODO generate here the system?
        //  this goes against the docs in LinearOperator: "Hence any pre-computations/caching should
        //  only be done in a lazy manner (e.g. during the first call of apply), and not in the
        //  constructor."
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
