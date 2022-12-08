#pragma once

#include "LinearOperator.h"

#include "Math.hpp"
#include "SphericalFunctionDescriptor.h"
#include "SphericalHarmonicsDescriptor.h"
#include "VolumeDescriptor.h"
namespace elsa
{
    /**
     * @brief Class for spherical harmonics transform.
     *
     * We always normalize the SphericalHarmoncis with respect to the measure given as
     * sphericalFunctionDesc->getWeights().sum()! Such that \f$ \Omega(S_2) \f$
     *
     * @author Max Endrass (endrass@cs.tum.edu), most boilerplate code courtesy of Matthias
     * Wieczorek
     * @author Matthias Wieczorek (wieczore@cs.tum.edu), logic merge with
     * XTTSphericalHarmonicsDescriptor and fixes for symmetry cases
     * @author Nikola Dinev (nikola.dinev@tum.de), port to elsa
     *
     * @tparam data_t real type
     */
    template <typename data_t = real_t>
    class SphericalHarmonicsTransform : public LinearOperator<data_t>
    {
    private:
        typedef LinearOperator<data_t> B;

    public:
        using VectorXd_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;
        using MatrixXd_t = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
        /**
         * @brief Constructor for SphericalHarmonicsTranform
         *
         * @param[in] domainDescriptor Descriptor for domain (original image or whatever)
         * @param[in] rangeDescriptor Descriptor for result range (max degree of SH polynomials
         * etc.)
         */
        SphericalHarmonicsTransform(const SphericalFunctionDescriptor<data_t>& domainDescriptor,
                                    const SphericalHarmonicsDescriptor& rangeDescriptor);

        virtual ~SphericalHarmonicsTransform() = default;

        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        void applyAdjointImpl(const DataContainer<data_t>& x,
                              DataContainer<data_t>& Atx) const override;

        SphericalHarmonicsTransform<data_t>* cloneImpl() const override;

        const MatrixXd_t& getInverseTransformationMatrix() const;

        MatrixXd_t getForwardTransformationMatrix() const;

    private:
        MatrixXd_t getSphericalHarmonicsBasis(
            const SphericalFunctionDescriptor<data_t>& sphericalFunctionDesc,
            const SphericalHarmonicsDescriptor& sphericalHarmonicsDesc);

        VolumeDescriptor
            getBasisDescriptor(const SphericalFunctionDescriptor<data_t>& sphericalFunctionDesc,
                               const SphericalHarmonicsDescriptor& sphericalHarmonicsDesc);
        VolumeDescriptor getBasisTransposeDescriptor(
            const SphericalFunctionDescriptor<data_t>& sphericalFunctionDesc,
            const SphericalHarmonicsDescriptor& sphericalHarmonicsDesc);

    protected:
        const VolumeDescriptor _basisDescriptor;
        const VolumeDescriptor _basisTransposeDescriptor;
        const MatrixXd_t _sphericalHarmonicsBasis;
    };
} // namespace elsa
