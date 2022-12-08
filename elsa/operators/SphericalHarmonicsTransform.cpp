#include "SphericalHarmonicsTransform.h"

namespace elsa
{

    template <typename data_t>
    SphericalHarmonicsTransform<data_t>::SphericalHarmonicsTransform(
        const SphericalFunctionDescriptor<data_t>& domainDescriptor,
        const SphericalHarmonicsDescriptor& rangeDescriptor)
        : B(domainDescriptor, rangeDescriptor),
          _basisDescriptor(getBasisDescriptor(domainDescriptor, rangeDescriptor)),
          _basisTransposeDescriptor(getBasisTransposeDescriptor(domainDescriptor, rangeDescriptor)),
          _sphericalHarmonicsBasis(getSphericalHarmonicsBasis(domainDescriptor, rangeDescriptor))
    {
    }

    template <typename data_t>
    SphericalHarmonicsTransform<data_t>* SphericalHarmonicsTransform<data_t>::cloneImpl() const
    {
        const auto& trueDomainDesc =
            static_cast<const SphericalFunctionDescriptor<data_t>&>(*(this->_domainDescriptor));
        const auto& trueRangeDesc =
            static_cast<const SphericalHarmonicsDescriptor&>(*(this->_domainDescriptor));
        return new SphericalHarmonicsTransform<data_t>(trueDomainDesc, trueRangeDesc);
    }

    template <typename data_t>
    void SphericalHarmonicsTransform<data_t>::applyAdjointImpl(const DataContainer<data_t>& x,
                                                               DataContainer<data_t>& Atx) const
    {

        try {
            // Range descriptor (coefficients that are to be reversed to scalar values on a sphere)
            const auto& trueRangeDesc =
                dynamic_cast<const SphericalHarmonicsDescriptor&>(x.getDataDescriptor());
            // Domain descriptor (for the DomainData where the (inversely) transformed values are to
            // be saved)
            const auto& trueDomainDesc =
                dynamic_cast<const SphericalFunctionDescriptor<data_t>&>(Atx.getDataDescriptor());
        } catch (const std::bad_cast& e) {
            throw std::logic_error(
                std::string("SphericalHarmonicsTransform::applyAdjoint: Expected a vector of "
                            "spherical harmonic coefficients")
                + "and a container for sampled function values as input.\nCheck the data "
                  "descriptors of the containers.");
        }

        index_t sliceCnt = _basisDescriptor.getNumberOfCoefficientsPerDimension()[1];
        auto tmp =
            DataContainer<data_t>(VolumeDescriptor{_basisDescriptor}, _sphericalHarmonicsBasis);
        Atx = tmp.slice(0) * x[0];
        for (index_t i = 1; i < sliceCnt; ++i) {
            Atx += tmp.slice(i) * x[i];
        }
    }

    template <typename data_t>
    void SphericalHarmonicsTransform<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                        DataContainer<data_t>& Ax) const
    {

        try {
            // This is the spherical function descriptor
            const auto& trueDomainDesc =
                dynamic_cast<const SphericalFunctionDescriptor<data_t>&>(x.getDataDescriptor());
            // This is the spherical harmonics basis descriptor (result)
            const auto& trueRangeDesc =
                dynamic_cast<const SphericalHarmonicsDescriptor&>(Ax.getDataDescriptor());

            // ** Reference: directSHT.m [MATLAB] **
            // Final result (multiply measured values ([theta,phi]-->R) and *transposed* (as opposed
            // to the Matlab reference and the thesis [easier to format in the thesis], anyway)
            // basis coefficients) MW: Changed this! We ALWAYS require the weights to be explicitly
            // be set!

            index_t sliceCnt = _basisTransposeDescriptor.getNumberOfCoefficientsPerDimension()[1];
            auto tmp = DataContainer<data_t>(VolumeDescriptor{_basisTransposeDescriptor},
                                             _sphericalHarmonicsBasis.transpose());
            auto weightsApplied =
                DataContainer<data_t>(
                    VolumeDescriptor{trueDomainDesc.getNumberOfCoefficientsPerDimension()},
                    trueDomainDesc.getWeights())
                * x;

            Ax = tmp.slice(0) * weightsApplied[0];
            for (index_t i = 1; i < sliceCnt; ++i) {
                Ax += tmp.slice(i) * weightsApplied[i];
            }
        } catch (const std::bad_cast& e) {
            throw std::logic_error(std::string("SphericalHarmonicsTransform::apply: Expected a "
                                               "vector of sampled function values ")
                                   + "and a container for spherical harmonic coefficients as "
                                     "input.\nCheck the data descriptors of the containers.");
        }
    }

    template <typename data_t>
    VolumeDescriptor SphericalHarmonicsTransform<data_t>::getBasisDescriptor(
        const SphericalFunctionDescriptor<data_t>& sphericalFunctionDesc,
        const SphericalHarmonicsDescriptor& sphericalHarmonicsDesc)
    {
        index_t samplingTuplesCount = sphericalFunctionDesc.getNumberOfCoefficients();
        index_t coeffColCount = sphericalHarmonicsDesc.getNumberOfCoefficients();
        return VolumeDescriptor{{samplingTuplesCount, coeffColCount}};
    }
    template <typename data_t>
    VolumeDescriptor SphericalHarmonicsTransform<data_t>::getBasisTransposeDescriptor(
        const SphericalFunctionDescriptor<data_t>& sphericalFunctionDesc,
        const SphericalHarmonicsDescriptor& sphericalHarmonicsDesc)
    {
        index_t samplingTuplesCount = sphericalFunctionDesc.getNumberOfCoefficients();
        index_t coeffColCount = sphericalHarmonicsDesc.getNumberOfCoefficients();
        return VolumeDescriptor{{coeffColCount, samplingTuplesCount}};
    }

    template <typename data_t>
    typename SphericalHarmonicsTransform<data_t>::MatrixXd_t
        SphericalHarmonicsTransform<data_t>::getSphericalHarmonicsBasis(
            const SphericalFunctionDescriptor<data_t>& sphericalFunctionDesc,
            const SphericalHarmonicsDescriptor& sphericalHarmonicsDesc)
    {

        index_t maxDegree = sphericalHarmonicsDesc.getMaxDegree();
        index_t samplingTuplesCount = sphericalFunctionDesc.getNumberOfCoefficients();
        index_t coeffColCount = sphericalHarmonicsDesc.getNumberOfCoefficients();
        MatrixXd_t basisCoeffs(samplingTuplesCount, coeffColCount);
        auto sym = sphericalHarmonicsDesc.getSymmetry();

        auto normFac =
            static_cast<data_t>(sqrt(4.0 * pi_t) / sqrt(sphericalFunctionDesc.getWeights().sum()));

        for (index_t i = 0; i < samplingTuplesCount; ++i) {
            index_t j = 0;

            auto dir = sphericalFunctionDesc.getIthDir(i);

            // theta: atan2 returns the elevation, so to get theta = the inclination, we need:
            // inclination = pi/2-elevation azimuth = phi the boost implementation provides
            // spherical harmonics which are normalized under the STANDARD L2 norm on the sphere,
            // i.e. \int_{\mathbb{S}_2} 1 d\Omega(u) = 4pi interpreting the sum of the weights of
            // sphericalFunctionDesc as the measure used on the sphere, we need to adjust the
            // normalization such that the L_2 norm under the used measure is 1 as we want
            // orthonormal spherical harmonics
            auto sh_dir = axdt::SH_basis_real(
                maxDegree, static_cast<data_t>((pi_t / 2.0) + atan2(dir[2], hypot(dir[0], dir[1]))),
                static_cast<data_t>(atan2(dir[1], dir[0])));

            for (int l = 0; l <= maxDegree; ++l) {
                for (int m = -l; m <= l; ++m) {
                    if ((sym == SphericalHarmonicsDescriptor::SYMMETRY::even && (l % 2) != 0)
                        || (sym == SphericalHarmonicsDescriptor::SYMMETRY::odd && (l % 2) == 0)) {
                        continue;
                    }

                    basisCoeffs(i, j) = normFac * sh_dir(l * l + l + m);
                    j++;
                }
            }
        }

        return basisCoeffs;
    }

    template <typename data_t>
    const typename SphericalHarmonicsTransform<data_t>::MatrixXd_t&
        SphericalHarmonicsTransform<data_t>::getInverseTransformationMatrix() const
    {
        return _sphericalHarmonicsBasis;
    }

    template <typename data_t>
    typename SphericalHarmonicsTransform<data_t>::MatrixXd_t
        SphericalHarmonicsTransform<data_t>::getForwardTransformationMatrix() const
    {

        auto& trueDomainDesc =
            static_cast<const SphericalFunctionDescriptor<data_t>&>(B::getDomainDescriptor());
        Eigen::DiagonalMatrix<data_t, Eigen::Dynamic> weights(
            trueDomainDesc.getNumberOfCoefficients());
        weights = Eigen::DiagonalMatrix<data_t, Eigen::Dynamic>(trueDomainDesc.getWeights());

        return _sphericalHarmonicsBasis.transpose() * weights;
    }

    // ----------------------------------------------
    // explicit template instantiation
    template class SphericalHarmonicsTransform<real_t>;
} // namespace elsa