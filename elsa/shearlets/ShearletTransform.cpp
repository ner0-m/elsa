#include "ShearletTransform.h"
#include "VolumeDescriptor.h"
#include "PartitionDescriptor.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    ShearletTransform<data_t>::ShearletTransform(index_t width, index_t height)
        : ShearletTransform(width, height, calculatejZero(width, height))
    {
    }

    template <typename data_t>
    ShearletTransform<data_t>::ShearletTransform(index_t width, index_t height, index_t jZero)
        : LinearOperator<data_t>(VolumeDescriptor{{width, height}},
                                 VolumeDescriptor{{calculateL(width, height), width, height}}),
          _width{width},
          _height{height},
          _jZero{jZero},
          _L{calculateL(width, height)}
    {
        if (width < 0 || height < 0) {
            throw LogicError("ShearletTransform: negative width/height were provided");
        }
        // TODO check for jZero < 0? probably yes
    }

    // TODO remove me before final MR, implement in some other place
    DataContainer<double> getReals(DataContainer<std::complex<double>> dc)
    {
        DataContainer<double> reals(dc.getDataDescriptor());
        for (index_t i = 0; i < dc.getSize(); i++) {
            reals[i] = dc[i].real();
        }
        return reals;
    }

    template <typename data_t>
    void ShearletTransform<data_t>::applyImpl(const DataContainer<data_t>& f,
                                              DataContainer<data_t>& SHf) const
    {
        Timer timeguard("ShearletTransform", "apply");

        if (_width != this->getDomainDescriptor().getNumberOfCoefficientsPerDimension()[0]
            || _height != this->getDomainDescriptor().getNumberOfCoefficientsPerDimension()[1]) {
            throw InvalidArgumentError("ShearletTransform: Width and height of the input do not "
                                       "match to that of this shearlet system");
        }

        Logger::get("ShearletTransform")
            ->info("Running the shearlet transform on a 2D signal of shape ({}, {}), on {} "
                   "scales with an oversampling factor of {} and {} spectra",
                   _width, _height, _jZero, _L,
                   _isSpectraComputed ? "precomputed" : "non-precomputed");

        if (!isSpectraComputed()) {
            computeSpectra();
        }

        // TODO use fft when available

        // DataContainer<std::complex<data_t>> fftImg = fft.fft2(f);
        // // AFAIK SHf's imaginary parts should all be 0 here, cast to float
        // SHf = getReals(fft.ifft2(getSpectra() * fftImg)); // element-wise product
    }

    template <typename data_t>
    void ShearletTransform<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                     DataContainer<data_t>& SHty) const
    {
        Timer timeguard("ShearletTransform", "applyAdjoint");

        if (_width != this->getDomainDescriptor().getNumberOfCoefficientsPerDimension()[0]
            || _height != this->getDomainDescriptor().getNumberOfCoefficientsPerDimension()[1]) {
            throw InvalidArgumentError("ShearletTransform: Width and height of the input do not "
                                       "match to that of this shearlet system");
        }

        Logger::get("ShearletTransform")
            ->info("Running the inverse shearlet transform on a 2D signal of shape ({}, {}), on {} "
                   "scales with an oversampling factor of {} and {} spectra",
                   _width, _height, _jZero, _L,
                   _isSpectraComputed ? "precomputed" : "non-precomputed");

        if (!isSpectraComputed()) {
            computeSpectra();
        }

        // TODO use fft when available

        // SHty = getReals(np.sum(fft.ifft2(fft.fft2(y) * getSpectra()), axis = 0));
    }

    // TODO consider simplifying the usage of floor/ceil
    // TODO consider ordering spectra based on the frequency domain, might be essential when
    //  training a DL model since it should learn these patterns based on a logical order
    // TODO consider utilizing OpenMP
    // TODO address casts
    // TODO consider accommodating for different type of shearing and parabolic scaling functions
    // TODO consider using [i]fftshift for even-sized signals
    // TODO consider adding various generating functions, e.g. smooth shearlets
    // TODO are we supporting real shearlets only? seems so
    template <typename data_t>
    void ShearletTransform<data_t>::computeSpectra() const
    {
        DataContainer<data_t> spectra(VolumeDescriptor{{_width, _height, _L}});
        const DataDescriptor& spectraDescr = spectra.getDataDescriptor();
        auto dims = spectraDescr.getNumberOfDimensions();
        auto sizeOfLastDim = spectraDescr.getNumberOfCoefficientsPerDimension()[dims - 1];
        auto sliceDesc = PartitionDescriptor(spectraDescr, sizeOfLastDim);
        spectra = 0; // is this mandatory? probably not if everything is replaced

        index_t i = 0;

        DataContainer<data_t> sectionZero(VolumeDescriptor{{_width, _height}});
        sectionZero = 0;
        for (auto w1 = static_cast<int>(-std::floor(_width / 2)); w1 < std::ceil(_width / 2);
             w1++) {
            for (auto w2 = static_cast<int>(-std::floor(_height / 2)); w2 < std::ceil(_height / 2);
                 w2++) {
                sectionZero(w1 < 0 ? w1 + _width : w1, w2 < 0 ? w2 + _height : w2) =
                    phiHat(w1 < 0 ? w1 + _width : w1, w2 < 0 ? w2 + _height : w2);
            }
        }
        spectra.slice(i) = sectionZero;
        i += 1;

        for (index_t j = 0; j < _jZero; j++) {
            for (auto k = static_cast<index_t>(-std::pow(2, j));
                 k <= static_cast<index_t>(std::pow(2, j)); k++) {
                DataContainer<data_t> sectionh(VolumeDescriptor{{_width, _height}});
                sectionh = 0;
                DataContainer<data_t> sectionv(VolumeDescriptor{{_width, _height}});
                sectionv = 0;
                DataContainer<data_t> sectionhxv(VolumeDescriptor{{_width, _height}});
                sectionhxv = 0;
                for (auto w1 = static_cast<int>(-std::floor(_width / 2));
                     w1 < static_cast<int>(std::ceil(_width / 2)); w1++) {
                    for (auto w2 = static_cast<int>(-std::floor(_height / 2));
                         w2 < static_cast<int>(std::ceil(_height / 2)); w2++) {
                        data_t horiz = 0;
                        data_t vertic = 0;
                        if (std::abs(w2) <= std::abs(w1)) {
                            horiz = psiHat(std::pow(4, -j) * w1,
                                           std::pow(4, -j) * k * w1 + std::pow(2, -j) * w2);
                        } else {
                            vertic = psiHat(std::pow(4, -j) * w2,
                                            std::pow(4, -j) * k * w2 + std::pow(2, -j) * w1);
                        }
                        if (std::abs(k) <= static_cast<index_t>(std::pow(2, j)) - 1) {
                            sectionh(w1 < 0 ? w1 + _width : w1, w2 < 0 ? w2 + _height : w2) = horiz;
                            sectionv(w1 < 0 ? w1 + _width : w1, w2 < 0 ? w2 + _height : w2) =
                                vertic;
                        } else if (std::abs(k) == static_cast<index_t>(std::pow(2, j))) {
                            sectionhxv(w1 < 0 ? w1 + _width : w1, w2 < 0 ? w2 + _height : w2) =
                                horiz + vertic;
                        }
                    }
                }
                // TODO AFAIK sectionh, sectionv, sectionhxv really should only have reals, double
                //  check
                if (std::abs(k) <= static_cast<index_t>(std::pow(2, j)) - 1) {
                    spectra.slice(i) = sectionh;
                    i += 1;
                    spectra.slice(i) = sectionv;
                    i += 1;
                } else if (std::abs(k) == static_cast<index_t>(std::pow(2, j))) {
                    spectra.slice(i) = sectionhxv;
                    i += 1;
                }
            }
        }

        _spectra = std::make_unique<DataContainer<data_t>>(spectra);

        _isSpectraComputed = true;
    }

    template <typename data_t>
    data_t ShearletTransform<data_t>::meyerFunction(data_t x) const
    {
        if (x < 0) {
            return 0;
        } else if (0 <= x && x <= 1) {
            return 35 * std::pow(x, 4) - 84 * std::pow(x, 5) + 70 * std::pow(x, 6)
                   - 20 * std::pow(x, 7);
        } else {
            return 1;
        }
    }

    template <typename data_t>
    data_t ShearletTransform<data_t>::b(data_t w) const
    {
        if (1 <= std::abs(w) && std::abs(w) <= 2) {
            return std::sin(pi<data_t> / 2 * meyerFunction(std::abs(w) - 1));
        } else if (2 < std::abs(w) && std::abs(w) <= 4) {
            return std::cos(pi<data_t> / 2 * meyerFunction(1 / 2 * std::abs(w) - 1));
        } else {
            return 0;
        }
    }

    template <typename data_t>
    data_t ShearletTransform<data_t>::phi(data_t w) const
    {
        if (std::abs(w) <= 1 / 2) {
            return 1;
        } else if (1 / 2 < std::abs(w) && std::abs(w) < 1) {
            return std::cos(pi<data_t> / 2 * meyerFunction(2 * std::abs(w) - 1));
        } else {
            return 0;
        }
    }

    template <typename data_t>
    data_t ShearletTransform<data_t>::phiHat(data_t w1, data_t w2) const
    {
        if (std::abs(w2) <= std::abs(w1)) {
            return phi(w1);
        } else {
            return phi(w2);
        }
    }

    template <typename data_t>
    data_t ShearletTransform<data_t>::psiHat1(data_t w) const
    {
        return std::sqrt(std::pow(b(2 * w), 2) + std::pow(b(w), 2));
    }

    template <typename data_t>
    data_t ShearletTransform<data_t>::psiHat2(data_t w) const
    {
        if (w <= 0) {
            return std::sqrt(meyerFunction(1 + w));
        } else {
            return std::sqrt(meyerFunction(1 - w));
        }
    }

    template <typename data_t>
    data_t ShearletTransform<data_t>::psiHat(data_t w1, data_t w2) const
    {
        if (w1 == 0) {
            return 0;
        } else {
            return psiHat1(w1) * psiHat2(w2 / w1);
        }
    }

    template <typename data_t>
    auto ShearletTransform<data_t>::getSpectra() const -> DataContainer<data_t>
    {
        return *_spectra;
    }

    template <typename data_t>
    bool ShearletTransform<data_t>::isSpectraComputed() const
    {
        return _isSpectraComputed;
    }

    template <typename data_t>
    index_t ShearletTransform<data_t>::calculatejZero(index_t width, index_t height)
    {
        return static_cast<index_t>(std::log2(std::max(width, height)) / 2);
    }

    template <typename data_t>
    index_t ShearletTransform<data_t>::calculateL(index_t width, index_t height)
    {
        return static_cast<index_t>(std::pow(2, (calculatejZero(width, height) + 2)) - 3);
    }

    template <typename data_t>
    ShearletTransform<data_t>* ShearletTransform<data_t>::cloneImpl() const
    {
        return new ShearletTransform<data_t>(_width, _height, _jZero);
    }

    template <typename data_t>
    bool ShearletTransform<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherST = downcast_safe<ShearletTransform>(&other);

        if (!otherST)
            return false;

        if (_width != otherST->_width)
            return false;

        if (_height != otherST->_height)
            return false;

        if (_jZero != otherST->_jZero)
            return false;

        return true;
    }

    template <typename data_t>
    auto ShearletTransform<data_t>::getWidth() const -> index_t
    {
        return _width;
    }

    template <typename data_t>
    auto ShearletTransform<data_t>::getHeight() const -> index_t
    {
        return _height;
    }

    template <typename data_t>
    auto ShearletTransform<data_t>::getL() const -> index_t
    {
        return _L;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ShearletTransform<float>;
    template class ShearletTransform<double>;
} // namespace elsa
