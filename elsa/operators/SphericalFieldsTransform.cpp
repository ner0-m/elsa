#include "SphericalFieldsTransform.h"

namespace elsa
{
    template <typename data_t>
    SphericalFieldsTransform<data_t>::SphericalFieldsTransform(
        const ParametrizedVolumeDescriptor& domainDescriptor,
        const ParametrizedVolumeDescriptor& rangeDescriptor)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor)
    {

        try {
            // get basis descriptor of domain
            const auto& sphericalFunctionDesc =
                dynamic_cast<const SphericalFunctionDescriptor<data_t>&>(
                    domainDescriptor.getBasisDescriptor());

            // get the descriptor for the final coordinate this may throw an exception if not set!
            const auto& sphericalHarmonicsDesc = dynamic_cast<const SphericalHarmonicsDescriptor&>(
                rangeDescriptor.getBasisDescriptor());

            _sht = std::make_unique<SphericalHarmonicsTransform<data_t>>(sphericalFunctionDesc,
                                                                         sphericalHarmonicsDesc);
        } catch (const std::bad_cast& e) {
            throw std::invalid_argument(
                std::string("SphericalFieldTransform: Domain descriptor should have a "
                            "SphericalFunctionDescriptor as basis\n")
                + "Range descriptor should have a SphericalHarmonicsDescriptor as basis.");
        }
    }

    template <typename data_t>
    void SphericalFieldsTransform<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                     DataContainer<data_t>& Ax) const
    {

        Timer timeguard("SphericalFieldsTransform", "apply()");

        const auto* trueDomainDesc =
            dynamic_cast<const ParametrizedVolumeDescriptor*>(&x.getDataDescriptor());
        const auto* sphericalFunctionDesc =
            dynamic_cast<const SphericalFunctionDescriptor<data_t>*>(
                &trueDomainDesc->getBasisDescriptor());

        const auto* trueRangeDesc =
            dynamic_cast<const ParametrizedVolumeDescriptor*>(&Ax.getDataDescriptor());
        const auto* sphericalHarmonicsDesc =
            dynamic_cast<const SphericalHarmonicsDescriptor*>(&trueRangeDesc->getBasisDescriptor());

        if ((trueDomainDesc == nullptr) || (trueRangeDesc == nullptr))
            throw std::logic_error("SphericalFieldsTransform::apply: Expected ParametrizedVolumes "
                                   "as input. Check the data descriptors of the containers.");

        if (sphericalFunctionDesc == nullptr)
            throw std::logic_error("SphericalFieldsTransform::apply: Operates on a volume of "
                                   "spherical functions. Check data descriptor.");

        if (sphericalHarmonicsDesc == nullptr)
            throw std::logic_error(
                "SphericalFieldsTransform::apply: Data container for output should be a container "
                "for spherical harmonic coefficients. Check basis descriptor.");

        if (trueDomainDesc->getNumberOfCoefficientsPerDimension()
            != this->_domainDescriptor->getNumberOfCoefficientsPerDimension())
            throw std::logic_error("SphericalFieldsTransform::apply: DataContainer x does not have "
                                   "the right dimensions.");

        if (trueRangeDesc->getNumberOfCoefficientsPerDimension()
            != this->_rangeDescriptor->getNumberOfCoefficientsPerDimension())
            throw std::logic_error("SphericalFieldsTransform::apply: Output DataContainer Ax does "
                                   "not have the right dimensions.");

        // get the degree
        index_t voxelCount = trueDomainDesc->getDescriptorOfBlock(0).getNumberOfCoefficients();
        index_t samplingDirs = sphericalFunctionDesc->getNumberOfCoefficients();
        index_t sphCoeffsCount = sphericalHarmonicsDesc->getNumberOfCoefficients();

        // transpose of mode-4 unfolding of x
        Eigen::Map<const RealMatrix_t> x4(&(x.storage()[0]), voxelCount, samplingDirs);

        // transpose of mode-4 unfolding of Ax
        Eigen::Map<RealMatrix_t> Ax4(&(Ax.storage()[0]), voxelCount, sphCoeffsCount);

        RealMatrix_t WVt = _sht->getForwardTransformationMatrix().transpose();

        // perform multiplication in chunks
        index_t step =
            voxelCount
            / trueDomainDesc->getDescriptorOfBlock(0).getNumberOfCoefficientsPerDimension()(0);
        for (index_t i = 0; i < voxelCount; i += step) {
            Ax4.middleRows(i, step) = x4.middleRows(i, step) * WVt;
        }
    }

    template <typename data_t>
    void SphericalFieldsTransform<data_t>::applyAdjointImpl(const DataContainer<data_t>& x,
                                                            DataContainer<data_t>& Atx) const
    {
        Timer timeguard("SphericalFieldsTransform", "applyAdjoint()");

        const auto* trueDomainDesc =
            dynamic_cast<const ParametrizedVolumeDescriptor*>(&Atx.getDataDescriptor());
        const auto* sphericalFunctionDesc =
            dynamic_cast<const SphericalFunctionDescriptor<data_t>*>(
                &trueDomainDesc->getBasisDescriptor());

        const auto* trueRangeDesc =
            dynamic_cast<const ParametrizedVolumeDescriptor*>(&x.getDataDescriptor());
        const auto* sphericalHarmonicsDesc =
            dynamic_cast<const SphericalHarmonicsDescriptor*>(&trueRangeDesc->getBasisDescriptor());

        if ((trueDomainDesc == nullptr) || (trueRangeDesc == nullptr))
            throw std::logic_error(
                "SphericalFieldsTransform::applyAdjoint: Expected ParametrizedVolumes as input. "
                "Check the data descriptors of the containers.");

        if (sphericalFunctionDesc == nullptr)
            throw std::logic_error("SphericalFieldsTransform::applyAdjoint: Operates on a volume "
                                   "of spherical harmonics coefficients. Check basis descriptor.");

        if (sphericalHarmonicsDesc == nullptr)
            throw std::logic_error(
                "SphericalFieldsTransform::applyAdjoint: Data container for output should be a "
                "container for spherical functions. Check basis descriptor.");

        if (trueDomainDesc->getNumberOfCoefficientsPerDimension()
            != B::_domainDescriptor->getNumberOfCoefficientsPerDimension())
            throw std::logic_error("SphericalFieldsTransform::applyAdjoint: Output DataContainer "
                                   "Atx does not have the right dimensions.");

        if (trueRangeDesc->getNumberOfCoefficientsPerDimension()
            != B::_rangeDescriptor->getNumberOfCoefficientsPerDimension())
            throw std::logic_error("SphericalFieldsTransform::applyAdjoint: DataContainer x does "
                                   "not have the right dimensions.");

        // get the degree
        index_t voxelCount = trueDomainDesc->getDescriptorOfBlock(0).getNumberOfCoefficients();
        index_t voxelCountFirstDim =
            trueDomainDesc->getDescriptorOfBlock(0).getNumberOfCoefficientsPerDimension()(0);
        index_t samplingDirs = sphericalFunctionDesc->getNumberOfCoefficients();
        index_t sphCoeffsCount = sphericalHarmonicsDesc->getNumberOfCoefficients();

        RealMatrix_t V = _sht->getInverseTransformationMatrix().transpose();

        // transpose of mode-4 unfolding of x
        Eigen::Map<const RealMatrix_t> x4(&(x.storage()[0]), voxelCount, sphCoeffsCount);
        // transpose of mode-4 unfolding of Atx
        Eigen::Map<RealMatrix_t> Atx4(&(Atx.storage()[0]), voxelCount, samplingDirs);

        // perform multiplication in chunks
        size_t step = voxelCount / voxelCountFirstDim;
        for (int i = 0; i < voxelCount; i += step) {
            Atx4.middleRows(i, step) = x4.middleRows(i, step) * V;
        }
    }

    template <typename data_t>
    SphericalFieldsTransform<data_t>* SphericalFieldsTransform<data_t>::cloneImpl() const
    {
        const auto& trueDomainDesc =
            static_cast<const ParametrizedVolumeDescriptor&>(this->getDomainDescriptor());
        const auto& trueRangeDesc =
            static_cast<const ParametrizedVolumeDescriptor&>(this->getDomainDescriptor());
        return new SphericalFieldsTransform<data_t>(trueDomainDesc, trueRangeDesc);
    }

    // ----------------------------------------------
    // explicit template instantiation
    template class SphericalFieldsTransform<real_t>;
} // namespace elsa