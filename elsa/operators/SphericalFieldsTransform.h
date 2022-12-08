#pragma once

#include "elsaDefines.h"
#include "LinearOperator.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "Timer.h"

#include "ParametrizedVolumeDescriptor.h"
#include "SphericalFunctionDescriptor.h"
#include "SphericalHarmonicsDescriptor.h"
#include "SphericalHarmonicsTransform.h"

namespace elsa
{
    /**
     * @brief Class for spherical harmonics transform for blocks.
     *
     * @author Max Endrass (endrass@cs.tum.edu), most boilerplate code courtesy of Matthias
     * Wieczorek
     * @author Matthias Wieczorek (wieczore@cs.tum.edu), logic merge and fixes
     * @author Nikola Dinev (nikola.dinev@tum.de), port to elsa
     *
     * @tparam real_t real type
     */
    template <typename data_t = real_t>
    class SphericalFieldsTransform : public LinearOperator<data_t>
    {
        using B = LinearOperator<data_t>;

    public:
        /**
         * @brief Constructor for SphericalFieldsTransform
         *
         * @param[in] domainDescriptor descriptor for reconstruction volume
         * @param[in] rangeDescriptor Descriptor for result range (max degree of SH polynomials
         * etc.)
         */
        SphericalFieldsTransform(const ParametrizedVolumeDescriptor& domainDescriptor,
                                 const ParametrizedVolumeDescriptor& rangeDescriptor);

        virtual ~SphericalFieldsTransform() = default;

        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        void applyAdjointImpl(const DataContainer<data_t>& x,
                              DataContainer<data_t>& Atx) const override;

        SphericalFieldsTransform<data_t>* cloneImpl() const override;

    protected:
        std::unique_ptr<SphericalHarmonicsTransform<data_t>> _sht;
    };
} // namespace elsa