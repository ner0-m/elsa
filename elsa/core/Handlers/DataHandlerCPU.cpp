#include "DataHandlerCPU.h"
#include "Error.h"
#include "TypeCasts.hpp"
#include "Assertions.hpp"

#include "DataDescriptor.h"

#include <iostream>

#if WITH_FFTW
#define EIGEN_FFTW_DEFAULT
#endif
#include <unsupported/Eigen/FFT>

namespace elsa
{
    namespace detail
    {
        /// Helper to check if the passed in handler is either an owning CPU handler or a map to one
        /// If it's one of them execute f1, else do f2
        template <typename Handler, typename FuncTrue, typename FuncFalse>
        auto mapIfCpuHandlerOrElse(const Handler& other, FuncTrue&& f1, FuncFalse f2)
        {
            using data_t = typename DataHandlerCPUTraits<Handler>::Scalar;

            if (is<DataHandlerCPU<data_t>>(other)) {
                const auto& y = downcast_safe<DataHandlerCPU<data_t>>(other);
                return f1(y);
            } else if (is<DataHandlerMapCPU<data_t>>(other)) {
                const auto& y = downcast_safe<DataHandlerMapCPU<data_t>>(other);
                return f1(y);
            }
            return f2();
        }
        // ------------------------------------------
        // DataHandlerCPUBase implementation
        // ------------------------------------------
        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::self() -> Derived&
        {
            return *static_cast<Derived*>(this);
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::self() const -> const Derived&
        {
            return *static_cast<const Derived*>(this);
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::write() -> DataMap_t
        {
            return self().write();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::read() const -> ConstDataMap_t
        {
            return self().read();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::getSize() const -> index_t
        {
            return read().size();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator[](index_t index) -> data_t&
        {
            return write()[index];
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator[](index_t index) const -> const data_t&
        {
            return read()[index];
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::dot(const DataHandler<data_t>& v) const -> data_t
        {
            ELSA_VERIFY(v.getSize() == getSize(),
                        "Dot product is only valid for handlers of equal size");

            // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback
            // version,
            // clang-format off
            return mapIfCpuHandlerOrElse(
                v,
                [&](auto&& y) { return read().dot(y.read()); },
                [&]() { return this->slowDotProduct(v); }
            );
            // clang-format on
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::squaredL2Norm() const -> GetFloatingPointType_t<data_t>
        {
            return read().squaredNorm();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::l2Norm() const -> GetFloatingPointType_t<data_t>
        {
            return read().norm();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::l0PseudoNorm() const -> index_t
        {
            using FloatType = GetFloatingPointType_t<data_t>;
            return (read().array().cwiseAbs() >= std::numeric_limits<FloatType>::epsilon()).count();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::l1Norm() const -> GetFloatingPointType_t<data_t>
        {
            return read().array().abs().sum();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::lInfNorm() const -> GetFloatingPointType_t<data_t>
        {
            return read().array().abs().maxCoeff();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::sum() const -> data_t
        {
            return read().sum();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::minElement() const -> data_t
        {
            if constexpr (isComplex<data_t>) {
                throw LogicError("DataHandlerCPU: minElement of complex type not supported");
            } else {
                return read().minCoeff();
            }
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::maxElement() const -> data_t
        {
            if constexpr (isComplex<data_t>) {
                throw LogicError("DataHandlerCPU: maxElement of complex type not supported");
            } else {
                return read().maxCoeff();
            }
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::fft(const DataDescriptor& source_desc, FFTNorm norm)
            -> DataHandler<data_t>&
        {
            base_fft<true>(source_desc, norm);
            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::ifft(const DataDescriptor& source_desc, FFTNorm norm)
            -> DataHandler<data_t>&
        {
            base_fft<false>(source_desc, norm);
            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator+=(const DataHandler<data_t>& v)
            -> DataHandler<data_t>&
        {
            ELSA_VERIFY(v.getSize() == getSize(),
                        "In-Place addition only valid with handlers of equal size");

            // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
            if (auto otherHandler = downcast_safe<DataHandlerCPUBase<Derived>>(&v)) {
                write() += otherHandler->read();
            } else {
                this->slowAddition(v);
            }

            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator-=(const DataHandler<data_t>& v)
            -> DataHandler<data_t>&
        {
            ELSA_VERIFY(v.getSize() == getSize(),
                        "In-Place subtraction only valid with handlers of equal size");

            // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
            if (auto otherHandler = downcast_safe<DataHandlerCPUBase<Derived>>(&v)) {
                write() -= otherHandler->read();
            } else {
                this->slowSubtraction(v);
            }

            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator*=(const DataHandler<data_t>& v)
            -> DataHandler<data_t>&
        {
            ELSA_VERIFY(v.getSize() == getSize(),
                        // TODO: Will this be caught by backward-cpp? Ensure it
                        "In-Place multiplication only valid with handlers of equal size");

            // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
            if (auto otherHandler = downcast_safe<DataHandlerCPUBase<Derived>>(&v)) {
                write().array() *= otherHandler->read().array();
            } else {
                this->slowMultiplication(v);
            }

            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator/=(const DataHandler<data_t>& v)
            -> DataHandler<data_t>&
        {
            ELSA_VERIFY(v.getSize() == getSize(),
                        "In-Place division only valid with handlers of equal size");

            // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
            if (auto otherHandler = downcast_safe<DataHandlerCPUBase<Derived>>(&v)) {
                write().array() /= otherHandler->read().array();
            } else {
                this->slowDivision(v);
            }

            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator+=(data_t scalar) -> DataHandler<data_t>&
        {
            write().array() += scalar;
            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator-=(data_t scalar) -> DataHandler<data_t>&
        {
            write().array() -= scalar;
            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator*=(data_t scalar) -> DataHandler<data_t>&
        {
            write().array() *= scalar;
            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator/=(data_t scalar) -> DataHandler<data_t>&
        {
            write().array() /= scalar;
            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::operator=(data_t scalar) -> DataHandler<data_t>&
        {
            write().setConstant(scalar);
            return *this;
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::assign(const DataHandler<data_t>& other) -> void
        {
            if (const auto otherHandler = downcast_safe<DataHandlerCPUBase<Derived>>(&other)) {
                write() = otherHandler->read();
            } else {
                this->slowAssign(other);
            }
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::assign(DataHandler<data_t>&& other) -> void
        {
            if (const auto otherHandler = downcast_safe<DataHandlerCPUBase<Derived>>(&other)) {
                write() = std::move(otherHandler->read());
            } else {
                this->slowAssign(other);
            }
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::isEqual(const DataHandler<data_t>& other) const -> bool
        {
            return mapIfCpuHandlerOrElse(
                other, [this](auto&& y) { return self() == y; }, []() { return false; });
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::accessData() const -> ConstDataMap_t
        {
            return read();
        }

        template <typename Derived>
        auto DataHandlerCPUBase<Derived>::accessData() -> DataMap_t
        {
            return write();
        }

        template <typename Derived>
        template <bool is_forward>
        auto DataHandlerCPUBase<Derived>::base_fft(const DataDescriptor& source_desc, FFTNorm norm)
            -> void
        {
            if constexpr (isComplex<data_t>) {
                const auto& src_shape = source_desc.getNumberOfCoefficientsPerDimension();
                const auto& src_dims = source_desc.getNumberOfDimensions();

                data_t* this_data = write().data();

                // TODO: fftw variant

                // generalization of an 1D-FFT
                // walk over each dimenson and 1d-fft one 'line' of data
                for (index_t dim_idx = 0; dim_idx < src_dims; ++dim_idx) {
                    // jumps in the data for the current dimension's data
                    // dim_size[0] * dim_size[1] * ...
                    // 1 for dim_idx == 0.
                    const index_t stride = src_shape.head(dim_idx).prod();

                    // number of coefficients for the current dimension
                    const index_t dim_size = src_shape(dim_idx);

                    // number of coefficients for the other dimensions
                    // this is the number of 1d-ffts we'll do
                    // e.g. shape=[2, 3, 4] and we do dim_idx=2 (=shape 4)
                    //   -> src_shape.prod() == 24 / 4 = 6 == 2*3
                    const index_t other_dims_size = src_shape.prod() / dim_size;

#ifndef EIGEN_FFTW_DEFAULT
// when using eigen+fftw, this corrupts the memory, so don't parallelize.
// error messages may include:
// * double free or corruption (fasttop)
// * malloc_consolidate(): unaligned fastbin chunk detected
#pragma omp parallel for
#endif
                    // do all the 1d-ffts along the current dimensions axis
                    for (index_t i = 0; i < other_dims_size; ++i) {

                        index_t ray_start = i;
                        // each time i is a multiple of stride,
                        // jump forward the current+previous dimensions' shape product
                        // (draw an indexed 3d cube to visualize this)
                        ray_start += (stride * (dim_size - 1)) * ((i - (i % stride)) / stride);

                        // this is one "ray" through the volume
                        Eigen::Map<Vector_t<data_t>, Eigen::AlignmentType::Unaligned,
                                   Eigen::InnerStride<>>
                            input_map(this_data + ray_start, dim_size,
                                      Eigen::InnerStride<>(stride));

                    using inner_t = GetFloatingPointType_t<typename DataVector_t::Scalar>;

                    Eigen::FFT<inner_t> fft_op;

                    // disable any scaling in eigen - normally it does 1/n for ifft
                    fft_op.SetFlag(Eigen::FFT<inner_t>::Flag::Unscaled);

                    Eigen::Matrix<std::complex<inner_t>, Eigen::Dynamic, 1> fft_in{dim_size};
                    Eigen::Matrix<std::complex<inner_t>, Eigen::Dynamic, 1> fft_out{dim_size};

                    // eigen internally copies the fwd input matrix anyway if
                    // it doesn't have stride == 1
                    fft_in =
                        input_map.block(0, 0, dim_size, 1).template cast<std::complex<inner_t>>();

                        if (unlikely(dim_size == 1)) {
                            // eigen kiss-fft crashes for size=1...
                            fft_out = fft_in;
                        } else {
                            // arguments for in and out _must not_ be the same matrix!
                            // they will corrupt wildly otherwise.
                            if constexpr (is_forward) {
                                fft_op.fwd(fft_out, fft_in);
                                if (norm == FFTNorm::FORWARD) {
                                    fft_out /= dim_size;
                                } else if (norm == FFTNorm::ORTHO) {
                                    fft_out /= std::sqrt(dim_size);
                                }
                            } else {
                                fft_op.inv(fft_out, fft_in);
                                if (norm == FFTNorm::BACKWARD) {
                                    fft_out /= dim_size;
                                } else if (norm == FFTNorm::ORTHO) {
                                    fft_out /= std::sqrt(dim_size);
                                }
                            }
                        }

                    // we can't directly use the map as fft output,
                    // since Eigen internally just uses the pointer to
                    // the map's first element, and doesn't respect stride at all..
                    input_map.block(0, 0, dim_size, 1) = fft_out.template cast<data_t>();
                }
            } else {
                throw Error{"fft with non-complex input container not supported"};
            }
        }
    } // namespace detail

    // ------------------------------------------
    // DataHandlerCPU implementation
    // ------------------------------------------

    template <typename data_t>
    auto DataHandlerCPU<data_t>::write() -> DataMap_t
    {
        return DataMap_t{data_.write().data(), data_.read().size()};
    }

    template <typename data_t>
    auto DataHandlerCPU<data_t>::read() const -> ConstDataMap_t
    {
        const auto mutable_data = const_cast<data_t*>(data_.read().data());
        return DataMap_t{mutable_data, data_.read().size()};
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(const DataHandlerCPU& other) : data_(other.data_)
    {
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(DataHandlerCPU&& other) noexcept
        : data_(std::move(other.data_))
    {
    }

    template <typename data_t>
    DataHandlerCPU<data_t>& DataHandlerCPU<data_t>::operator=(const DataHandlerCPU& other)
    {
        return *this = DataHandlerCPU(other);
    }

    template <typename data_t>
    DataHandlerCPU<data_t>& DataHandlerCPU<data_t>::operator=(DataHandlerCPU&& other) noexcept
    {
        auto tmp = std::move(other);
        swap(*this, tmp);
        return *this;
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(index_t size) : data_(size)
    {
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(Vector_t<data_t> vec) : data_(vec)
    {
    }

    template <typename data_t>
    auto DataHandlerCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements)
        -> std::unique_ptr<DataHandler<data_t>>
    {
        if (startIndex >= this->getSize() || numberOfElements > this->getSize() - startIndex)
            throw InvalidArgumentError("DataHandler: requested block out of bounds");

        return std::make_unique<DataHandlerMapCPU<data_t>>(
            DataHandlerMapCPU{this, startIndex, startIndex + numberOfElements});
    }

    template <typename data_t>
    auto DataHandlerCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements) const
        -> std::unique_ptr<const DataHandler<data_t>>
    {
        if (startIndex >= this->getSize() || numberOfElements > this->getSize() - startIndex)
            throw InvalidArgumentError("DataHandler: requested block out of bounds");

        // NOTE: Save as a const version is returned, but maybe there is a nicer way?
        auto* mutable_this = const_cast<DataHandlerCPU<data_t>*>(this);
        return std::make_unique<const DataHandlerMapCPU<data_t>>(
            DataHandlerMapCPU{mutable_this, startIndex, startIndex + numberOfElements});
    }

    // ------------------------------------------
    // DataHandlerMapCPU implementation
    // ------------------------------------------

    template <typename data_t>
    auto DataHandlerMapCPU<data_t>::write() -> DataMap_t
    {
        auto data = ptr_->write().data();
        auto beginning = data + start_;
        const auto size = end_ - start_;

        return DataMap_t{beginning, size};
    }

    template <typename data_t>
    auto DataHandlerMapCPU<data_t>::read() const -> ConstDataMap_t
    {
        // TODO: Clean this up, I don't want a const cast here
        const auto data = const_cast<data_t*>(ptr_->read().data());
        const auto beginning = data + start_;
        const auto size = end_ - start_;

        return DataMap_t{beginning, size};
    }

    template <typename data_t>
    DataHandlerMapCPU<data_t>::DataHandlerMapCPU(const DataHandlerMapCPU& other)
        : ptr_(other.ptr_), start_(other.start_), end_(other.end_)
    {
    }

    template <typename data_t>
    DataHandlerMapCPU<data_t>::DataHandlerMapCPU(DataHandlerMapCPU&& other) noexcept
        : ptr_(other.ptr_), start_(other.start_), end_(other.end_)
    {
    }

    template <typename data_t>
    DataHandlerMapCPU<data_t>& DataHandlerMapCPU<data_t>::operator=(const DataHandlerMapCPU& other)
    {
        return *this = DataHandlerMapCPU(other);
    }

    template <typename data_t>
    DataHandlerMapCPU<data_t>&
        DataHandlerMapCPU<data_t>::operator=(DataHandlerMapCPU&& other) noexcept
    {
        auto tmp = std::move(other);
        swap(*this, tmp);
        return *this;
    }

    template <typename data_t>
    auto DataHandlerMapCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements)
        -> std::unique_ptr<DataHandler<data_t>>
    {
        return ptr_->getBlock(start_ + startIndex, numberOfElements);
    }

    template <typename data_t>
    auto DataHandlerMapCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements) const
        -> std::unique_ptr<const DataHandler<data_t>>
    {
        return ptr_->getBlock(start_ + startIndex, numberOfElements);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataHandlerCPU<float>;
    template class DataHandlerCPU<complex<float>>;
    template class DataHandlerCPU<double>;
    template class DataHandlerCPU<complex<double>>;
    template class DataHandlerCPU<index_t>;

    template class DataHandlerMapCPU<float>;
    template class DataHandlerMapCPU<std::complex<float>>;
    template class DataHandlerMapCPU<double>;
    template class DataHandlerMapCPU<std::complex<double>>;
    template class DataHandlerMapCPU<index_t>;

    namespace detail
    {
        template class DataHandlerCPUBase<DataHandlerCPU<float>>;
        template class DataHandlerCPUBase<DataHandlerCPU<double>>;
        template class DataHandlerCPUBase<DataHandlerCPU<std::complex<float>>>;
        template class DataHandlerCPUBase<DataHandlerCPU<std::complex<double>>>;

        template class DataHandlerCPUBase<DataHandlerMapCPU<float>>;
        template class DataHandlerCPUBase<DataHandlerMapCPU<double>>;
        template class DataHandlerCPUBase<DataHandlerMapCPU<std::complex<float>>>;
        template class DataHandlerCPUBase<DataHandlerMapCPU<std::complex<double>>>;
    } // namespace detail
} // namespace elsa
