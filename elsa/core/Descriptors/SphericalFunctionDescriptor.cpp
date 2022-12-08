#include "SphericalFunctionDescriptor.h"

namespace elsa
{

    template <typename data_t>
    SphericalFunctionDescriptor<data_t>::SphericalFunctionDescriptor(const DirVecList& dirs,
                                                                     const WeightVec& weights)
        : DataDescriptor(dirs.size() * IndexVector_t::Ones(1)), _dirs(dirs), _weights(weights)
    {
        if (dirs.size() != static_cast<size_t>(weights.size()))
            throw std::invalid_argument(
                "SphericalFunctionDescriptor: Sizes of directions list and weights do not match.");

        // just to make sure they are normalized ;)
        for (DirVec& dir : _dirs)
            dir.normalize();
    }

    template <typename data_t>
    SphericalFunctionDescriptor<data_t>::SphericalFunctionDescriptor(const DirVecList& dirs)
        : DataDescriptor(dirs.size() * IndexVector_t::Ones(1)),
          _dirs(dirs),
          _weights(WeightVec::Ones(dirs.size()))
    {
        // just to make sure they are normalized ;)
        for (DirVec& dir : _dirs)
            dir.normalize();
    }

    template <typename data_t>
    SphericalFunctionDescriptor<data_t>* SphericalFunctionDescriptor<data_t>::cloneImpl() const
    {
        return new SphericalFunctionDescriptor<data_t>(getDirs(), getWeights());
    }

    // ----------------------------------------------
    // explicit template instantiation
    template class SphericalFunctionDescriptor<real_t>;

} // namespace elsa
