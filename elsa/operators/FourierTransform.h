#pragma once

#include <complex>
#include "LinearOperator.h"

namespace elsa
{
    /**
     * \brief Operator for applying multi-dimensional fourier transforms.
     *
     * \author Jonas Jelten - initial code
     *
     * \tparam data_t data type for the domain and range of the transformation,
     *                defaulting to real_t
     *
     * Implements the n-dimensional signal fourier transformation.
     * Can support multiple backends, by default uses Eigen::FFT with FFTW.
     */
    template <typename data_t = std::complex<real_t>>
    class FourierTransform : public LinearOperator<data_t> {
    private:
        using B = LinearOperator<data_t>;

        /** working data container for the fft.
            like in datacontainer, we operate on the vector in n dimensions. */
        using fftvector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;
    public:
        /**
         * \brief create a fourier transform operator
         *
         * \param[in] domainDescriptor metadata defining the domain and range of the transformation
         */
        explicit FourierTransform(const DataDescriptor& domainDescriptor);

        ~FourierTransform() override = default;

    protected:
        /**
         * \brief perform the fourier transformation
         * \param x inputData (image matrix)
         * \param Ax outputData (fourier transformed image matrix)
         */
        void applyImpl(const DataContainer<data_t> &x, DataContainer<data_t> &Ax) const override;

        /**
         * \brief TODO ifft
         * \param x inputData (XXX)
         * \param Atx outputData (XXX)
         */
        void applyAdjointImpl(const DataContainer<data_t> &y, DataContainer<data_t> &Aty) const override;

        /// implement the polymorphic clone operation
        FourierTransform *cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const B &other) const override;

        /// recursively called fft implementation
        void fft(const fftvector_t &in,
                 fftvector_t &out,
                 index_t dims) const;

        void fft1d(const fftvector_t &in,
                   fftvector_t& out) const;

        void ifft1d(const fftvector_t &in,
                    fftvector_t& out) const;

    };

} // namespace elsa
