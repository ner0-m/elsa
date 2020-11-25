#include "FourierTransform.h"

#include "Timer.h"

#include <unsupported/Eigen/FFT>


namespace elsa {
    template <typename data_t>
    FourierTransform<data_t>::FourierTransform(const DataDescriptor &domainDescriptor)
            : B(domainDescriptor, domainDescriptor) {
    }

    template <typename data_t>
    void FourierTransform<data_t>::applyImpl(
        const DataContainer<data_t> &x,
        DataContainer<data_t> &Ax) const {

        Timer<> timeguard("FourierTransform", "apply()");

        auto x_size = x.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        auto x_dims = x.getDataDescriptor().getNumberOfDimensions();

        // input container size must match dimensionality of operator setup
        assert(x_size.size() == this->_domainDescriptor.getNumberOfDimensions());

        fftvector_t input{x.getSize()};
        input.setZero();

        // TODO: avoid this copy, instead access datahandler or get an eigen-matrix directly
        // convert to datacontainer to eigen-matrix
        // another approach: move fft implementation to datahandler
        for (index_t i = 0; i < x.getSize(); ++i) {
            input(i) = x(i);
        }

        fftvector_t output(x.getSize());
        output.setZero();

        this->fft(input, output, x_dims);

        for (index_t i = 0; i < output.size(); ++i) {
            Ax(i) = output(i);
        }
    }

    template <typename data_t>
    void FourierTransform<data_t>::fft(
        const fftvector_t &in,
        fftvector_t &out,
        index_t dims) const {

        auto dim_idx = dims - 1;

        if (dim_idx == 0) {
            this->fft1d(in, out);
        /*
        } else if (dim_idx == 1) {
            // TODO: use fftw 2d transformation directly!
        */
        } else {
            const auto &in_coeffs_per_dim = this->_domainDescriptor->getNumberOfCoefficientsPerDimension();
            const auto &out_coeffs_per_dim = this->_rangeDescriptor->getNumberOfCoefficientsPerDimension();

            const index_t in_stride = in_coeffs_per_dim.head(dim_idx).prod();
            const index_t out_stride = out_coeffs_per_dim.head(dim_idx).prod();

            // number of coefficients for the current dimension
            const index_t dim_size = out_coeffs_per_dim(dim_idx);

            // number of coefficients for the other dimensions
            // e.g. [10x10x10], and dim_idx=2 -> out.size() == 1000 -> /10 = 100
            const index_t dims_remaining_size = out.size() / dim_size;

            // TODO: in 2d-case, no need to copy
            #pragma omp parallel for
            for (index_t i = 0; i < dim_size; ++i) {
                fftvector_t in_tmp(in_stride);
                for (index_t j = 0; j < in_stride; j++) {
                    in_tmp(j) = in[in_stride * i + j];
                }

                fftvector_t out_tmp(out_stride);
                out_tmp.setZero();

                // recursive call!
                fft(in_tmp, out_tmp, dims - 1);

                // store out_stride elements at output row i
                out.segment(i * out_stride, out_stride) = out_tmp;
            }

            #pragma omp parallel for
            for (index_t i = 0; i < dims_remaining_size; ++i) {
                // for the column-calculations,
                // map the correct indices, i.e. use InnerStride
                // to specify the discance between two consecutive indices

                // yes, use out as input, since out contains the
                // calculation of the rows from the above loop.
                const Eigen::Map<fftvector_t, 0, Eigen::InnerStride<>>
                        input_map(const_cast<data_t*>(out.data() + i),
                                  dim_size,
                                  Eigen::InnerStride<>(out_stride));

                Eigen::Map<fftvector_t, 0, Eigen::InnerStride<>>
                        output_map(out.data() + i,
                                   dim_size,
                                   Eigen::InnerStride<>(out_stride));

                // copy the data into a new vector
                const fftvector_t in_tmp(input_map);
                fftvector_t out_tmp(output_map);

                this->fft1d(in_tmp, out_tmp);

                // bring temporary results into the output vector
                // the map is just holding pointers to the output vector
                // update happens implicitly
                output_map = out_tmp;
            }
        }
    }

    template <typename data_t>
    void FourierTransform<data_t>::applyAdjointImpl(const DataContainer<data_t>& x,
                                                    DataContainer<data_t>& Atx) const
    {
        Timer<> timeguard("FourierTransform", "applyAdjoint()");

        // TODO ifft
    }

    template <typename data_t>
    FourierTransform<data_t>* FourierTransform<data_t>::cloneImpl() const
    {
        auto& domainDescriptor = static_cast<const DataDescriptor&>(*this->_domainDescriptor);

        return new FourierTransform(domainDescriptor);
    }

    template <typename data_t>
    bool FourierTransform<data_t>::isEqual(const B& other) const
    {
        if (!B::isEqual(other))
            return false;

        auto otherOP = dynamic_cast<const FourierTransform *>(&other);
        if (!otherOP)
            return false;

        // TODO actually check for equality!
        return true;
    }

    template <typename data_t>
    void FourierTransform<data_t>::fft1d(const fftvector_t &in,
                                         fftvector_t &out) const {
        Eigen::FFT<typename data_t::value_type> fft;
        fft.fwd(out, in);
    }


    template <typename data_t>
    void FourierTransform<data_t>::ifft1d(const fftvector_t &in,
                                            fftvector_t &out) const {
        Eigen::FFT<typename data_t::value_type> fft;
        fft.inv(out, in);
    }

    template class FourierTransform<std::complex<float>>;
    template class FourierTransform<std::complex<double>>;
} // namespace elsa
