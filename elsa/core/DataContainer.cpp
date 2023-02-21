#include "DataContainer.h"
#include "DataContainerFormatter.hpp"
#include "FormatConfig.h"
#include "BlockDescriptor.h"
#include "RandomBlocksDescriptor.h"
#include "PartitionDescriptor.h"
#include "Error.h"
#include "TypeCasts.hpp"
#include "Assertions.h"

#include "Complex.h"

#include "Functions.hpp"
#include "elsaDefines.h"
#include "functions/Conj.hpp"
#include "functions/Imag.hpp"
#include "functions/Real.hpp"

#include "reductions/DotProduct.h"
#include "reductions/L0.h"
#include "reductions/L1.h"
#include "reductions/L2.h"
#include "reductions/LInf.h"
#include "reductions/Sum.h"
#include "reductions/Extrema.h"

#include "transforms/Absolute.h"
#include "transforms/Add.h"
#include "transforms/Assign.h"
#include "transforms/Clip.h"
#include "transforms/Cast.h"
#include "transforms/Sub.h"
#include "transforms/Div.h"
#include "transforms/Extrema.h"
#include "transforms/InplaceAdd.h"
#include "transforms/InplaceSub.h"
#include "transforms/InplaceMul.h"
#include "transforms/InplaceDiv.h"
#include "transforms/Square.h"
#include "transforms/Sqrt.h"
#include "transforms/Log.h"
#include "transforms/Lincomb.h"
#include "transforms/Exp.h"
#include "transforms/Imag.h"
#include "transforms/Real.h"

#include <utility>
#include <cmath>
#include <algorithm>

#if WITH_FFTW
#define EIGEN_FFTW_DEFAULT
#endif
#include <unsupported/Eigen/FFT>
namespace elsa
{
    // TODO: Move this somewhere elsa!
    namespace detail
    {
        template <class data_t, bool is_forward>
        void fftImpl(data_t* this_data, const IndexVector_t& src_shape, index_t src_dims,
                     FFTNorm norm)
        {
            using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

            if constexpr (isComplex<data_t>) {
                // TODO: fftw variant

                // generalization of an 1D-FFT
                // walk over each dimension and 1d-fft one 'line' of data
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
                        Eigen::Map<DataVector_t, Eigen::AlignmentType::Unaligned,
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
                        fft_in = input_map.block(0, 0, dim_size, 1)
                                     .template cast<std::complex<inner_t>>();

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
                }
            } else {
                throw Error{"fft with non-complex input container not supported"};
            }
        }
    } // namespace detail

    template <class data_t>
    void fft(ContiguousStorage<data_t>& x, const DataDescriptor& desc, FFTNorm norm)
    {
        const auto& src_shape = desc.getNumberOfCoefficientsPerDimension();
        const auto& src_dims = desc.getNumberOfDimensions();

        detail::fftImpl<data_t, true>(x.data().get(), src_shape, src_dims, norm);
    }

    template <class data_t>
    void ifft(ContiguousStorage<data_t>& x, const DataDescriptor& desc, FFTNorm norm)
    {
        const auto& src_shape = desc.getNumberOfCoefficientsPerDimension();
        const auto& src_dims = desc.getNumberOfDimensions();

        detail::fftImpl<data_t, false>(x.data().get(), src_shape, src_dims, norm);
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor& dataDescriptor)
        : _dataDescriptor{dataDescriptor.clone()},
          storage_{
              ContiguousStorage<data_t>(asUnsigned(_dataDescriptor->getNumberOfCoefficients()))}
    {
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor& dataDescriptor,
                                         const Eigen::Matrix<data_t, Eigen::Dynamic, 1>& data)
        : _dataDescriptor{dataDescriptor.clone()},
          storage_{std::in_place_type<ContiguousStorage<data_t>>, data.begin(), data.end()}
    {
        if (getSize() != dataDescriptor.getNumberOfCoefficients())
            throw InvalidArgumentError("DataContainer: initialization vector has invalid size");
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor& dataDescriptor,
                                         const ContiguousStorage<data_t>& storage)
        : _dataDescriptor{dataDescriptor.clone()}, storage_{storage}
    {
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor& dataDescriptor,
                                         ContiguousStorageView<data_t> span)
        : _dataDescriptor{dataDescriptor.clone()}, storage_{span}
    {
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataContainer<data_t>& other)
        : _dataDescriptor{other._dataDescriptor->clone()}, storage_{other.storage_}
    {
    }

    template <typename data_t>
    DataContainer<add_complex_t<data_t>> DataContainer<data_t>::asComplex() const
    {
        return elsa::asComplex(*this);
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(const DataContainer<data_t>& other)
    {
        if (this != &other) {
            _dataDescriptor = other._dataDescriptor->clone();

            // Assign the values from other to this storage, if this is a view, this will not
            // reallocate, but write through to the original data container
            std::visit(
                overloaded{
                    [](auto& self, const auto other) { self.assign(other.begin(), other.end()); },
                },
                storage_, other.storage_);
        }

        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(DataContainer<data_t>&& other) noexcept
        : _dataDescriptor{std::move(other._dataDescriptor)}, storage_{std::move(other.storage_)}
    {
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(DataContainer<data_t>&& other) noexcept
    {
        _dataDescriptor = std::move(other._dataDescriptor);

        // If this is a view, we need to write the values from other into this, this is due to the
        // dual requirement of data containers to be owning and views
        if (isView()) {
            std::visit(
                overloaded{
                    [](auto& self, const auto other) { self.assign(other.begin(), other.end()); },
                },
                storage_, other.storage_);

        } else {
            storage_ = std::move(other.storage_);
        }

        return *this;
    }

    template <typename data_t>
    const DataDescriptor& DataContainer<data_t>::getDataDescriptor() const
    {
        return *_dataDescriptor;
    }

    template <typename data_t>
    bool DataContainer<data_t>::isOwning() const
    {
        return std::visit(
            overloaded{[](const auto& storage) {
                return std::is_same_v<std::decay_t<decltype(storage)>, ContiguousStorage<data_t>>;
            }},
            storage_);
    }

    template <typename data_t>
    bool DataContainer<data_t>::isView() const
    {
        return std::visit(overloaded{[](const auto& storage) {
                              return std::is_same_v<std::decay_t<decltype(storage)>,
                                                    ContiguousStorageView<data_t>>;
                          }},
                          storage_);
    }

    template <typename data_t>
    const ContiguousStorage<data_t>& DataContainer<data_t>::storage() const
    {
        using RetRef = const ContiguousStorage<data_t>&;
        return std::visit(
            overloaded{
                [](const ContiguousStorage<data_t>& storage) -> RetRef { return storage; },
                [](ContiguousStorageView<data_t> storage) -> RetRef { return storage.storage(); }},
            storage_);
    }

    template <typename data_t>
    ContiguousStorage<data_t>& DataContainer<data_t>::storage()
    {
        using RetRef = ContiguousStorage<data_t>&;
        return std::visit(
            overloaded{
                [](ContiguousStorage<data_t>& storage) -> RetRef { return storage; },
                [](ContiguousStorageView<data_t> storage) -> RetRef { return storage.storage(); }},
            storage_);
    }

    template <typename data_t>
    index_t DataContainer<data_t>::getSize() const
    {
        return std::visit(
            overloaded{
                [](const auto& storage) { return asSigned(storage.size()); },
            },
            storage_);
    }

    template <typename data_t>
    typename DataContainer<data_t>::reference DataContainer<data_t>::operator[](index_t index)
    {
        ELSA_VERIFY(index >= 0);
        ELSA_VERIFY(index < getSize());

        return std::visit(
            overloaded{
                [index](auto& storage) -> reference { return storage[asUnsigned(index)]; },
            },
            storage_);
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_reference
        DataContainer<data_t>::operator[](index_t index) const
    {
        // ELSA_VERIFY(index >= 0);
        // ELSA_VERIFY(index < getSize());

        return std::visit(
            overloaded{
                [index](const auto& storage) -> const_reference {
                    return storage[asUnsigned(index)];
                },
            },
            storage_);
    }

    template <typename data_t>
    data_t DataContainer<data_t>::at(const IndexVector_t& coordinate) const
    {
        const auto arr = coordinate.array();
        if ((arr < 0).any()
            || (arr >= _dataDescriptor->getNumberOfCoefficientsPerDimension().array()).any()) {
            return 0;
        }

        return (*this)[_dataDescriptor->getIndexFromCoordinate(coordinate)];
    }

    template <typename data_t>
    typename DataContainer<data_t>::reference
        DataContainer<data_t>::operator()(const IndexVector_t& coordinate)
    {
        // const auto arr = coordinate.array();
        // const auto shape = _dataDescriptor->getNumberOfCoefficientsPerDimension().array();
        // ELSA_VERIFY((arr >= 0).all());
        // ELSA_VERIFY((arr < shape).all());

        return (*this)[_dataDescriptor->getIndexFromCoordinate(coordinate)];
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_reference
        DataContainer<data_t>::operator()(const IndexVector_t& coordinate) const
    {
        // const auto arr = coordinate.array();
        // const auto shape = _dataDescriptor->getNumberOfCoefficientsPerDimension().array();
        // ELSA_VERIFY((arr >= 0).all());
        // ELSA_VERIFY((arr < shape).all());

        return (*this)[_dataDescriptor->getIndexFromCoordinate(coordinate)];
    }

    template <typename data_t>
    data_t DataContainer<data_t>::dot(const DataContainer<data_t>& other) const
    {
        return elsa::dot(begin(), end(), other.begin());
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t>::squaredL2Norm() const
    {
        return elsa::squaredL2Norm(begin(), end());
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t>::l2Norm() const
    {
        return elsa::l2Norm(begin(), end());
    }

    template <typename data_t>
    index_t DataContainer<data_t>::l0PseudoNorm() const
    {
        return elsa::l0PseudoNorm(begin(), end());
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t>::l1Norm() const
    {
        return elsa::l1Norm(begin(), end());
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t>::lInfNorm() const
    {
        return elsa::lInf(begin(), end());
    }

    template <typename data_t>
    data_t DataContainer<data_t>::sum() const
    {
        return elsa::sum(begin(), end());
    }

    template <typename data_t>
    data_t DataContainer<data_t>::minElement() const
    {
        return elsa::minElement(begin(), end());
    }

    template <typename data_t>
    data_t DataContainer<data_t>::maxElement() const
    {
        return elsa::maxElement(begin(), end());
    }

    template <typename data_t>
    void DataContainer<data_t>::fft(FFTNorm norm)
    {
        std::visit(overloaded{[&](ContiguousStorage<data_t>& storage) {
                                  elsa::fft(storage, *_dataDescriptor, norm);
                              },
                              [&](ContiguousStorageView<data_t>& storage) {
                                  elsa::fft(storage.storage(), *_dataDescriptor, norm);
                              }},
                   storage_);
    }

    template <typename data_t>
    void DataContainer<data_t>::ifft(FFTNorm norm)
    {
        std::visit(overloaded{[&](ContiguousStorage<data_t>& storage) {
                                  elsa::ifft(storage, *_dataDescriptor, norm);
                              },
                              [&](ContiguousStorageView<data_t>& storage) {
                                  elsa::ifft(storage.storage(), *_dataDescriptor, norm);
                              }},
                   storage_);
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator+=(const DataContainer<data_t>& dc)
    {
        elsa::inplaceAdd(begin(), end(), dc.begin());
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator-=(const DataContainer<data_t>& dc)
    {
        elsa::inplaceSub(begin(), end(), dc.begin());
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator*=(const DataContainer<data_t>& dc)
    {
        elsa::inplaceMul(begin(), end(), dc.begin());
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator/=(const DataContainer<data_t>& dc)
    {
        elsa::inplaceDiv(begin(), end(), dc.begin());
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator+=(data_t scalar)
    {
        elsa::inplaceAddScalar(begin(), end(), scalar);
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator-=(data_t scalar)
    {
        elsa::inplaceSubScalar(begin(), end(), scalar);
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator*=(data_t scalar)
    {
        elsa::inplaceMulScalar(begin(), end(), scalar);
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator/=(data_t scalar)
    {
        elsa::inplaceDivScalar(begin(), end(), scalar);
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(data_t scalar)
    {
        elsa::fill(begin(), end(), scalar);
        return *this;
    }

    template <typename data_t>
    bool DataContainer<data_t>::operator==(const DataContainer<data_t>& other) const
    {
        if (*_dataDescriptor != *other._dataDescriptor)
            return false;

        // if (*_dataHandler != *other._dataHandler)
        //     return false;

        return true;
    }

    template <typename data_t>
    bool DataContainer<data_t>::operator!=(const DataContainer<data_t>& other) const
    {
        return !(*this == other);
    }

    template <typename data_t>
    DataContainer<data_t> DataContainer<data_t>::getBlock(index_t i)
    {
        const auto blockDesc = downcast_safe<BlockDescriptor>(_dataDescriptor.get());
        if (!blockDesc)
            throw LogicError("DataContainer: cannot get block from not-blocked container");

        if (i >= blockDesc->getNumberOfBlocks() || i < 0)
            throw InvalidArgumentError("DataContainer: block index out of bounds");

        size_t startIndex = asUnsigned(blockDesc->getOffsetOfBlock(i));
        const auto& ithDesc = blockDesc->getDescriptorOfBlock(i);
        size_t blockSize = asUnsigned(ithDesc.getNumberOfCoefficients());

        return std::visit(overloaded{[&](ContiguousStorage<data_t>& storage) {
                                         auto span = ContiguousStorageView<data_t>(
                                             storage, startIndex, startIndex + blockSize);
                                         return DataContainer<data_t>{ithDesc, span};
                                     },
                                     [&](ContiguousStorageView<data_t> storage) {
                                         auto span = ContiguousStorageView<data_t>(
                                             storage.storage(), startIndex, startIndex + blockSize);
                                         return DataContainer<data_t>{ithDesc, span};
                                     }},
                          storage_);
    }

    template <typename data_t>
    const DataContainer<data_t> DataContainer<data_t>::getBlock(index_t i) const
    {
        const auto blockDesc = downcast_safe<BlockDescriptor>(_dataDescriptor.get());
        if (!blockDesc)
            throw LogicError("DataContainer: cannot get block from not-blocked container");

        if (i >= blockDesc->getNumberOfBlocks() || i < 0)
            throw InvalidArgumentError("DataContainer: block index out of bounds");

        size_t startIndex = asUnsigned(blockDesc->getOffsetOfBlock(i));
        const auto& ithDesc = blockDesc->getDescriptorOfBlock(i);
        size_t blockSize = asUnsigned(ithDesc.getNumberOfCoefficients());

        return std::visit(overloaded{[&](const ContiguousStorage<data_t>& storage) {
                                         auto span = ContiguousStorageView<data_t>(
                                             // Casting const away is okay, as we return a const
                                             // container
                                             const_cast<ContiguousStorage<data_t>&>(storage),
                                             startIndex, startIndex + blockSize);
                                         return DataContainer<data_t>{ithDesc, span};
                                     },
                                     [&](ContiguousStorageView<data_t> storage) {
                                         auto span = ContiguousStorageView<data_t>(
                                             storage.storage(), startIndex, startIndex + blockSize);
                                         return DataContainer<data_t>{ithDesc, span};
                                     }},
                          storage_);
    }

    template <typename data_t>
    DataContainer<data_t> DataContainer<data_t>::viewAs(const DataDescriptor& dataDescriptor)
    {
        if (dataDescriptor.getNumberOfCoefficients() != getSize())
            throw InvalidArgumentError("DataContainer: view must have same size as container");

        return std::visit(overloaded{[&](ContiguousStorage<data_t>& storage) {
                                         auto span = ContiguousStorageView<data_t>(
                                             storage, 0,
                                             asUnsigned(dataDescriptor.getNumberOfCoefficients()));
                                         return DataContainer<data_t>{dataDescriptor, span};
                                     },
                                     [&](ContiguousStorageView<data_t> storage) {
                                         return DataContainer<data_t>{dataDescriptor, storage};
                                     }},
                          storage_);
    }

    template <typename data_t>
    const DataContainer<data_t>
        DataContainer<data_t>::viewAs(const DataDescriptor& dataDescriptor) const
    {
        if (dataDescriptor.getNumberOfCoefficients() != getSize())
            throw InvalidArgumentError("DataContainer: view must have same size as container");

        return std::visit(overloaded{[&](const ContiguousStorage<data_t>& storage) {
                                         auto span = ContiguousStorageView<data_t>(
                                             const_cast<ContiguousStorage<data_t>&>(storage), 0,
                                             asUnsigned(dataDescriptor.getNumberOfCoefficients()));
                                         return DataContainer<data_t>{dataDescriptor, span};
                                     },
                                     [&](ContiguousStorageView<data_t> storage) {
                                         return DataContainer<data_t>{dataDescriptor, storage};
                                     }},
                          storage_);
    }

    template <typename data_t>
    const DataContainer<data_t> DataContainer<data_t>::slice(index_t i) const
    {
        auto& desc = getDataDescriptor();
        auto dim = desc.getNumberOfDimensions();
        auto sizeOfLastDim = desc.getNumberOfCoefficientsPerDimension()[dim - 1];

        if (i >= sizeOfLastDim) {
            throw LogicError("Trying to access out of bound slice");
        }

        if (sizeOfLastDim == 1) {
            return *this;
        }

        auto sliceDesc = PartitionDescriptor(desc, sizeOfLastDim);

        // Now set the slice
        return viewAs(sliceDesc).getBlock(i);
    }

    template <typename data_t>
    DataContainer<data_t> DataContainer<data_t>::slice(index_t i)
    {
        auto& desc = getDataDescriptor();
        auto dim = desc.getNumberOfDimensions();
        auto sizeOfLastDim = desc.getNumberOfCoefficientsPerDimension()[dim - 1];

        if (i >= sizeOfLastDim) {
            throw LogicError("Trying to access out of bound slice");
        }

        if (sizeOfLastDim == 1) {
            return *this;
        }

        auto sliceDesc = PartitionDescriptor(desc, sizeOfLastDim);

        // Now set the slice
        return viewAs(sliceDesc).getBlock(i);
    }

    template <typename data_t>
    typename DataContainer<data_t>::iterator DataContainer<data_t>::begin()
    {
        return std::visit(
            overloaded{
                [](auto& storage) { return storage.begin(); },
            },
            storage_);
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_iterator DataContainer<data_t>::begin() const
    {
        return cbegin();
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_iterator DataContainer<data_t>::cbegin() const
    {
        return std::visit(
            overloaded{
                [](const auto& storage) { return storage.cbegin(); },
            },
            storage_);
    }

    template <typename data_t>
    typename DataContainer<data_t>::iterator DataContainer<data_t>::end()
    {
        return std::visit(
            overloaded{
                [](auto& storage) { return storage.end(); },
            },
            storage_);
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_iterator DataContainer<data_t>::end() const
    {
        return cend();
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_iterator DataContainer<data_t>::cend() const
    {
        return std::visit(
            overloaded{
                [](const auto& storage) { return storage.cend(); },
            },
            storage_);
    }

    template <typename data_t>
    void DataContainer<data_t>::format(std::ostream& os, format_config cfg) const
    {
        DataContainerFormatter<data_t> fmt{cfg};
        fmt.format(os, *this);
    }

    template <typename data_t>
    DataContainer<data_t> concatenate(const DataContainer<data_t>& dc1,
                                      const DataContainer<data_t>& dc2)
    {
        auto desc1 = dc1.getDataDescriptor().clone();
        auto desc2 = dc2.getDataDescriptor().clone();

        if (desc1->getNumberOfDimensions() != desc2->getNumberOfDimensions()) {
            throw LogicError("Can't concatenate two DataContainers with different dimension");
        }

        std::vector<std::unique_ptr<DataDescriptor>> descriptors;
        descriptors.reserve(2);
        descriptors.push_back(std::move(desc1));
        descriptors.push_back(std::move(desc2));

        auto blockDesc = RandomBlocksDescriptor(descriptors);
        auto concatenated = DataContainer<data_t>(blockDesc);

        concatenated.getBlock(0) = dc1;
        concatenated.getBlock(1) = dc2;
        return concatenated;
    }

    template <typename data_t>
    DataContainer<data_t> fftShift2D(const DataContainer<data_t>& dc)
    {
        assert(dc.getDataDescriptor().getNumberOfDimensions() == 2
               && "DataContainer::fftShift2D: currently only supporting 2D signals");

        const DataDescriptor& dataDescriptor = dc.getDataDescriptor();
        IndexVector_t numOfCoeffsPerDim = dataDescriptor.getNumberOfCoefficientsPerDimension();
        index_t m = numOfCoeffsPerDim[0];
        index_t n = numOfCoeffsPerDim[1];

        index_t firstShift = m / 2;
        index_t secondShift = n / 2;

        DataContainer<data_t> copyDC(dataDescriptor);

        for (index_t i = 0; i < m; ++i) {
            for (index_t j = 0; j < n; ++j) {
                copyDC((i + firstShift) % m, (j + secondShift) % n) = dc(i, j);
            }
        }

        return copyDC;
    }

    template <typename data_t>
    DataContainer<data_t> ifftShift2D(const DataContainer<data_t>& dc)
    {
        assert(dc.getDataDescriptor().getNumberOfDimensions() == 2
               && "DataContainer::ifftShift2D: currently only supporting 2D signals");

        const DataDescriptor& dataDescriptor = dc.getDataDescriptor();
        IndexVector_t numOfCoeffsPerDim = dataDescriptor.getNumberOfCoefficientsPerDimension();
        index_t m = numOfCoeffsPerDim[0];
        index_t n = numOfCoeffsPerDim[1];

        index_t firstShift = -m / 2;
        index_t secondShift = -n / 2;

        DataContainer<data_t> copyDC(dataDescriptor);

        for (index_t i = 0; i < m; ++i) {
            for (index_t j = 0; j < n; ++j) {
                index_t leftIndex = (((i + firstShift) % m) + m) % m;
                index_t rightIndex = (((j + secondShift) % n) + n) % n;
                copyDC(leftIndex, rightIndex) = dc(i, j);
            }
        }

        return copyDC;
    }

    template <typename data_t>
    DataContainer<data_t> clip(const DataContainer<data_t>& dc, data_t min, data_t max)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        elsa::clip(dc.begin(), dc.end(), copy.begin(), min, max);
        return copy;
    }

    template <typename data_t>
    DataContainer<data_t> exp(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        elsa::exp(dc.begin(), dc.end(), copy.begin());
        return copy;
    }

    template <typename data_t>
    DataContainer<data_t> log(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        elsa::log(dc.begin(), dc.end(), copy.begin());
        return copy;
    }

    template <typename data_t>
    DataContainer<data_t> square(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        elsa::square(dc.begin(), dc.end(), copy.begin());
        return copy;
    }

    template <typename data_t>
    DataContainer<data_t> sqrt(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        elsa::sqrt(dc.begin(), dc.end(), copy.begin());
        return copy;
    }

    template <typename data_t>
    DataContainer<data_t> minimum(const DataContainer<data_t>& dc, SelfType_t<data_t> scalar)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        elsa::minimum(dc.begin(), dc.end(), scalar, copy.begin());
        return copy;
    }

    template <typename data_t>
    DataContainer<data_t> maximum(const DataContainer<data_t>& dc, SelfType_t<data_t> scalar)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        elsa::maximum(dc.begin(), dc.end(), scalar, copy.begin());
        return copy;
    }

    template <class data_t>
    DataContainer<data_t> materialize(const DataContainer<data_t>& x)
    {
        if (x.isOwning()) {
            return x;
        } else {
            ContiguousStorage<data_t> storage(x.begin(), x.end());
            return DataContainer(x.getDataDescriptor(), storage);
        }
    }

    template <typename data_t>
    DataContainer<value_type_of_t<data_t>> cwiseAbs(const DataContainer<data_t>& dc)
    {
        using T = GetFloatingPointType_t<data_t>;
        DataContainer<T> copy(dc.getDataDescriptor());

        elsa::cwiseAbs(dc.begin(), dc.end(), copy.begin());
        return copy;
    }

    template <typename data_t>
    DataContainer<value_type_of_t<data_t>> real(const DataContainer<data_t>& dc)
    {
        DataContainer<value_type_of_t<data_t>> result(dc.getDataDescriptor());
        elsa::real(dc.begin(), dc.end(), result.begin());
        return result;
    }

    template <typename data_t>
    DataContainer<value_type_of_t<data_t>> imag(const DataContainer<data_t>& dc)
    {
        DataContainer<value_type_of_t<data_t>> result(dc.getDataDescriptor());
        elsa::imag(dc.begin(), dc.end(), result.begin());
        return result;
    }

    template <typename data_t>
    DataContainer<add_complex_t<data_t>> asComplex(const DataContainer<data_t>& dc)
    {
        if constexpr (isComplex<data_t>) {
            return dc;
        } else {
            DataContainer<complex<data_t>> ret{dc.getDataDescriptor()};

            // extend with complex zero value
            elsa::cast(dc.begin(), dc.end(), ret.begin());
            return ret;
        }
    }

    template <typename data_t>
    DataContainer<data_t> operator-(const DataContainer<data_t>& lhs,
                                    const DataContainer<data_t>& rhs)
    {
        // TODO: Do size checking!
        DataContainer<data_t> ret{lhs.getDataDescriptor()};
        elsa::sub(lhs.begin(), lhs.end(), rhs.begin(), ret.begin());
        return ret;
    }

    template <typename data_t, typename Scalar, typename>
    DataContainer<std::common_type_t<data_t, Scalar>> operator-(const DataContainer<data_t>& dc,
                                                                const Scalar& s)
    {
        using T = std::common_type_t<data_t, Scalar>;

        DataContainer<T> ret{dc.getDataDescriptor()};
        elsa::subScalar(dc.begin(), dc.end(), T(s), ret.begin());
        return ret;
    }

    template <typename Scalar, typename data_t, typename>
    DataContainer<std::common_type_t<Scalar, data_t>> operator-(const Scalar& s,
                                                                const DataContainer<data_t>& dc)
    {
        using T = std::common_type_t<Scalar, data_t>;

        DataContainer<T> ret{dc.getDataDescriptor()};
        elsa::subScalar(T(s), dc.begin(), dc.end(), ret.begin());
        return ret;
    }

    template <typename data_t>
    DataContainer<data_t> operator/(const DataContainer<data_t>& lhs,
                                    const DataContainer<data_t>& rhs)
    {
        DataContainer<data_t> ret{lhs.getDataDescriptor()};
        elsa::div(lhs.begin(), lhs.end(), rhs.begin(), ret.begin());
        return ret;
    }

    template <typename data_t, typename Scalar, typename>
    DataContainer<std::common_type_t<data_t, Scalar>> operator/(const DataContainer<data_t>& dc,
                                                                const Scalar& s)
    {
        using T = std::common_type_t<data_t, Scalar>;

        DataContainer<T> ret{dc.getDataDescriptor()};
        elsa::divScalar(dc.begin(), dc.end(), T(s), ret.begin());
        return ret;
    }

    template <typename Scalar, typename data_t, typename>
    DataContainer<std::common_type_t<Scalar, data_t>> operator/(const Scalar& s,
                                                                const DataContainer<data_t>& dc)
    {
        using T = std::common_type_t<Scalar, data_t>;

        DataContainer<T> ret{dc.getDataDescriptor()};
        elsa::divScalar(T(s), dc.begin(), dc.end(), ret.begin());
        return ret;
    }

    template <typename xdata_t, typename ydata_t>
    DataContainer<value_type_of_t<std::common_type_t<xdata_t, ydata_t>>>
        cwiseMax(const DataContainer<xdata_t>& lhs, const DataContainer<ydata_t>& rhs)
    {
        using data_t = value_type_of_t<std::common_type_t<xdata_t, ydata_t>>;

        DataContainer<data_t> copy(rhs.getDataDescriptor());
        elsa::cwiseMax(lhs.begin(), lhs.end(), rhs.begin(), copy.begin());
        return copy;
    }

    template <typename xdata_t, typename ydata_t>
    DataContainer<value_type_of_t<std::common_type_t<xdata_t, ydata_t>>>
        cwiseMin(const DataContainer<xdata_t>& lhs, const DataContainer<ydata_t>& rhs)
    {
        using data_t = value_type_of_t<std::common_type_t<xdata_t, ydata_t>>;

        DataContainer<data_t> copy(rhs.getDataDescriptor());
        elsa::cwiseMin(lhs.begin(), lhs.end(), rhs.begin(), copy.begin());
        return copy;
    }

    template <class data_t>
    DataContainer<data_t> lincomb(SelfType_t<data_t> a, const DataContainer<data_t>& x,
                                  SelfType_t<data_t> b, const DataContainer<data_t>& y)
    {
        if (x.getDataDescriptor() != y.getDataDescriptor()) {
            throw InvalidArgumentError("lincomb: x and y are of different size");
        }

        auto out = DataContainer<data_t>(x.getDataDescriptor());
        lincomb(a, x, b, y, out);
        return out;
    }

    template <class data_t>
    void lincomb(SelfType_t<data_t> a, const DataContainer<data_t>& x, SelfType_t<data_t> b,
                 const DataContainer<data_t>& y, DataContainer<data_t>& out)
    {
        if (x.getDataDescriptor() != y.getDataDescriptor()) {
            throw InvalidArgumentError("lincomb: x and y are of different size");
        }

        if (x.getDataDescriptor() != out.getDataDescriptor()) {
            throw InvalidArgumentError("lincomb: input and output vectors are of different size");
        }

        lincomb(a, x.begin(), x.end(), b, y.begin(), out.begin());
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataContainer<float>;
    template class DataContainer<complex<float>>;
    template class DataContainer<double>;
    template class DataContainer<complex<double>>;
    template class DataContainer<index_t>;

    template DataContainer<float> clip<float>(const DataContainer<float>& dc, float min, float max);
    template DataContainer<double> clip<double>(const DataContainer<double>& dc, double min,
                                                double max);

    template DataContainer<float> concatenate<float>(const DataContainer<float>&,
                                                     const DataContainer<float>&);
    template DataContainer<double> concatenate<double>(const DataContainer<double>&,
                                                       const DataContainer<double>&);
    template DataContainer<complex<float>>
        concatenate<complex<float>>(const DataContainer<complex<float>>&,
                                    const DataContainer<complex<float>>&);
    template DataContainer<complex<double>>
        concatenate<complex<double>>(const DataContainer<complex<double>>&,
                                     const DataContainer<complex<double>>&);

    template void lincomb<float>(SelfType_t<float>, const DataContainer<float>&, SelfType_t<float>,
                                 const DataContainer<float>&, DataContainer<float>&);
    template void lincomb<double>(SelfType_t<double>, const DataContainer<double>&,
                                  SelfType_t<double>, const DataContainer<double>&,
                                  DataContainer<double>&);
    template DataContainer<float> lincomb<float>(SelfType_t<float>, const DataContainer<float>&,
                                                 SelfType_t<float>, const DataContainer<float>&);
    template DataContainer<double> lincomb<double>(SelfType_t<double>, const DataContainer<double>&,
                                                   SelfType_t<double>,
                                                   const DataContainer<double>&);

#define ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET(fn, type) \
    template DataContainer<value_type_of_t<type>> fn<type>(const DataContainer<type>&);

#define ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET_TYPES(fn)        \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET(fn, index_t);        \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET(fn, float);          \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET(fn, double);         \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET(fn, complex<float>); \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET(fn, complex<double>);

    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET_TYPES(cwiseAbs)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET_TYPES(real)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET_TYPES(imag)

#undef ELSA_INSTANTIATE_UNARY_TRANSFORMATION_REAL_RET

#define ELSA_INSTANTIATE_UNARY_TRANSFORMATION(fn, type) \
    template DataContainer<type> fn<type>(const DataContainer<type>&);

#define ELSA_INSTANTIATE_UNARY_TRANSFORMATION_TYPES(fn)       \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION(fn, index_t)        \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION(fn, float)          \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION(fn, double)         \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION(fn, complex<float>) \
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION(fn, complex<double>)

    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_TYPES(fftShift2D)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_TYPES(ifftShift2D)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_TYPES(exp)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_TYPES(log)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_TYPES(square)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_TYPES(sqrt)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_TYPES(materialize)

#undef ELSA_INSTANTIATE_UNARY_TRANSFORMATION

#define ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(fn, type) \
    template DataContainer<type> fn<type>(const DataContainer<type>&, SelfType_t<type>);

    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(minimum, index_t)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(minimum, float)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(minimum, double)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(minimum, complex<float>)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(minimum, complex<double>)

    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(maximum, index_t)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(maximum, float)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(maximum, double)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(maximum, complex<float>)
    ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX(maximum, complex<double>)

#undef ELSA_INSTANTIATE_UNARY_TRANSFORMATION_MINMAX

#define ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(fn, type)                             \
    template DataContainer<value_type_of_t<std::common_type_t<type, type>>> fn<type, type>( \
        const DataContainer<type>&, const DataContainer<type>&);

#define ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(fn, type1, type2)                          \
    template DataContainer<value_type_of_t<std::common_type_t<type1, type2>>> fn<type1, type2>( \
        const DataContainer<type1>&, const DataContainer<type2>&);                              \
    template DataContainer<value_type_of_t<std::common_type_t<type2, type1>>> fn<type2, type1>( \
        const DataContainer<type2>&, const DataContainer<type1>&);

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMax, index_t)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMax, float)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMax, double)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMax, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMax, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, index_t, float)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, index_t, double)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, index_t, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, index_t, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, float, double)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, float, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, float, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, double, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, double, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMax, thrust::complex<float>,
                                                 thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMin, index_t)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMin, float)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMin, double)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMin, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_SINGLE(cwiseMin, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, index_t, float)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, index_t, double)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, index_t, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, index_t, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, float, double)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, float, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, float, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, double, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, double, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED(cwiseMin, thrust::complex<float>,
                                                 thrust::complex<double>)

#undef ELSA_INSTANTIATE_BINARY_TRANSFORMATION_MIXED

#define ELSA_INSTANTIATE_BINARY_TRANSFORMATION(fn, type) \
    template DataContainer<type> fn<type>(const DataContainer<type>&, const DataContainer<type>&);

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator-, index_t)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator-, float)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator-, double)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator-, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator-, thrust::complex<double>)

    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator/, index_t)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator/, float)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator/, double)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator/, thrust::complex<float>)
    ELSA_INSTANTIATE_BINARY_TRANSFORMATION(operator/, thrust::complex<double>)

#undef ELSA_INSTANTIATE_BINARY_TRANSFORMATION

#define ELSA_INSTANTIATE_OPERATOR_MIXED(fn, dtype, stype)                            \
    template DataContainer<std::common_type_t<dtype, stype>> fn<dtype, stype, void>( \
        const DataContainer<dtype>&, const stype&);                                  \
    template DataContainer<std::common_type_t<stype, dtype>> fn<stype, dtype, void>( \
        const stype&, const DataContainer<dtype>&);

#define ELSA_INSTANTIATE_OPERATOR_MIXED_ALL(fn)                                          \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, index_t, int)                                    \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, index_t, index_t)                                \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, index_t, float)                                  \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, index_t, double)                                 \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, index_t, thrust::complex<float>)                 \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, index_t, thrust::complex<double>)                \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, float, int)                                      \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, float, index_t)                                  \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, float, float)                                    \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, float, double)                                   \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, float, thrust::complex<float>)                   \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, float, thrust::complex<double>)                  \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, double, int)                                     \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, double, index_t)                                 \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, double, float)                                   \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, double, double)                                  \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, double, thrust::complex<float>)                  \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, double, thrust::complex<double>)                 \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<float>, int)                     \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<float>, index_t)                 \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<float>, float)                   \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<float>, double)                  \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<float>, thrust::complex<float>)  \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<float>, thrust::complex<double>) \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<double>, int)                    \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<double>, index_t)                \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<double>, float)                  \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<double>, double)                 \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<double>, thrust::complex<float>) \
    ELSA_INSTANTIATE_OPERATOR_MIXED(fn, thrust::complex<double>, thrust::complex<double>)

    ELSA_INSTANTIATE_OPERATOR_MIXED_ALL(operator/)
    ELSA_INSTANTIATE_OPERATOR_MIXED_ALL(operator-)

#define ELSA_INSTANTIATE_AS_COMPLEX(type) \
    template DataContainer<add_complex_t<type>> asComplex<type>(const DataContainer<type>&);

    ELSA_INSTANTIATE_AS_COMPLEX(index_t)
    ELSA_INSTANTIATE_AS_COMPLEX(float)
    ELSA_INSTANTIATE_AS_COMPLEX(double)
    ELSA_INSTANTIATE_AS_COMPLEX(complex<float>)
    ELSA_INSTANTIATE_AS_COMPLEX(complex<double>)

#undef ELSA_INSTANTIATE_AS_COMPLEX
} // namespace elsa
