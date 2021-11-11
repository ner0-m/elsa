#pragma once

#include <complex>
#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Operator for applying multi-dimensional fourier transforms.
     *
     * @author Jonas Jelten - initial code
     *
     * @tparam data_t data type for the domain and range of the transformation,
     *                defaulting to real_t
     *
     * Implements the n-dimensional signal fourier transformation.
     * Can support multiple backends, by default uses Eigen::FFT with FFTW.
     */
    template <typename data_t = std::complex<real_t>>
    class FourierTransform : public LinearOperator<data_t>
    {
    private:
        using B = LinearOperator<data_t>;

        /** working data container for the fft.
            like in datacontainer, we operate on the vector in n dimensions. */
        using fftvector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    public:
        /**
         * @brief create a fourier transform operator
         *
         * @param[in] domainDescriptor metadata defining the domain and range of the transformation
         * @param[in] norm metadata indicating which forward/inverse transform is scaled and
         * on which of the predefined normalization factors
         */
        explicit FourierTransform(const DataDescriptor& domainDescriptor,
                                  FFTNorm norm = FFTNorm::BACKWARD);

        ~FourierTransform() override = default;

    protected:
        /**
         * @brief perform the fourier transformation
         * @param x inputData (image matrix)
         * @param Ax outputData (fourier transformed image matrix)
         */
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /**
         * @brief perform the inverse fourier transformation
         * @param x inputData (image matrix in frequency domain)
         * @param Atx outputData (inversely fourier transformed image matrix)
         */
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        FourierTransform* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const B& other) const override;

    private:
        FFTNorm norm;
    };

} // namespace elsa
