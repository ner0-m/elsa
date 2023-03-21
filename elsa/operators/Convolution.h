#pragma once

#include <complex>
#include "DataContainer.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Operator for convolving multi-dimensional measurements.
     *
     * This implements the deterministic Matrix+Kernel (de)convolutions.
     * It's not suitable for deconvolving probabilistic errored measurements,
     * for them, use a specific problem solver.
     *
     * The Type::FFT implements the convolution using fourier transforms:
     * `(A * B)` is the same as `IFFT(FFT(A) x FFT(B))`
     * TODO: http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c13-1.pdf
     * TODO https://github.com/scipy/scipy/blob/v1.10.1/scipy/signal/_signaltools.py#L556-L671
     * TODO def oaconvolve(in1, in2, mode="full", axes=None):
     *      Convolve two N-dimensional arrays using the overlap-add method.
     *
     * @author Jonas Jelten - initial code
     *
     * @tparam data_t data type for the domain and range of the transformation,
     *                defaulting to real_t
     *
     * Implements the n-dimensional signal convolution with a given kernel.
     */
    template <typename data_t = complex<real_t>>
    class Convolution : public LinearOperator<data_t>
    {
    private:
        using B = LinearOperator<data_t>;

    public:
        /**
         * supported convolution implementation types.
         */
        enum class Type {
            FFT,
        };

        /**
         * @brief create a convolution operator
         *
         * @param[in] domainDescriptor metadata defining the domain and range of the convolution
         * @param[in] kernel to convolve with
         */
        Convolution(const DataDescriptor& domainDescriptor, const DataContainer<data_t>& kernel,
                    Type method = Type::FFT);

        ~Convolution() override = default;

    protected:
        /**
         * @brief perform the deterministic convolution
         * @param x inputData (image matrix)
         * @param Ax outputData (convolved image matrix)
         */
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /**
         * @brief perform the deconvolution.
         *        this is not really the adjoint, but rather the inverse.
         *        don't use this if the input data has noise! you'll need a
         *        deconvolution solver instead.
         * @param x inputData (image matrix)
         * @param Atx outputData (deconvolved image matrix)
         */
        void applyAdjointImpl(const DataContainer<data_t>& x,
                              DataContainer<data_t>& Atx) const override;

        /// implement the polymorphic clone operation
        Convolution* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const B& other) const override;

    private:
        /// which convolution method to use
        Type method;

        ConvolutionFilter<data_t> kernel;
    };

} // namespace elsa
