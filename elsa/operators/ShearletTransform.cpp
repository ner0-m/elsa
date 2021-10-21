#include "ShearletTransform.h"
#include "FourierTransform.h"
#include "VolumeDescriptor.h"
#include "Timer.h"

namespace elsa
{
    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>::ShearletTransform(IndexVector_t spatialDimensions)
        : ShearletTransform(spatialDimensions[0], spatialDimensions[1])
    {
        if (spatialDimensions.size() != 2) {
            throw LogicError("ShearletTransform: a non-2D shape of input was provided");
        }
    }

    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>::ShearletTransform(index_t width, index_t height)
        : LinearOperator<ret_t>(
            VolumeDescriptor{{width, height}},
            VolumeDescriptor{{width, height, calculateNumOfLayers(width, height)}}),
          _width{width},
          _height{height},
          _numOfScales{calculateNumOfScales(width, height)},
          _numOfLayers{calculateNumOfLayers(width, height)}
    {
        if (width < 0 || height < 0) {
            throw LogicError("ShearletTransform: negative width/height were provided");
        }
    }

    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>::ShearletTransform(index_t width, index_t height,
                                                        index_t numOfScales)
        : LinearOperator<ret_t>(
            VolumeDescriptor{{width, height}},
            VolumeDescriptor{{width, height, calculateNumOfLayers(numOfScales)}}),
          _width{width},
          _height{height},
          _numOfScales{numOfScales},
          _numOfLayers{calculateNumOfLayers(numOfScales)}
    {
        if (width < 0 || height < 0) {
            throw LogicError("ShearletTransform: negative width/height were provided");
        }
        if (numOfScales < 0) {
            throw LogicError("ShearletTransform: negative number of scales was provided");
        }
    }

    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>::ShearletTransform(
        index_t width, index_t height, index_t numOfScales,
        std::optional<DataContainer<data_t>> spectra)
        : LinearOperator<ret_t>(
            VolumeDescriptor{{width, height}},
            VolumeDescriptor{{width, height, calculateNumOfLayers(numOfScales)}}),
          _spectra{spectra},
          _isSpectraComputed{_spectra.has_value()},
          _width{width},
          _height{height},
          _numOfScales{numOfScales},
          _numOfLayers{calculateNumOfLayers(numOfScales)}
    {
        if (width < 0 || height < 0) {
            throw LogicError("ShearletTransform: negative width/height were provided");
        }
        if (numOfScales < 0) {
            throw LogicError("ShearletTransform: negative number of scales was provided");
        }
    }

    // TODO implement sumByAxis in DataContainer and remove me
    template <typename ret_t, typename data_t>
    DataContainer<std::complex<data_t>> ShearletTransform<ret_t, data_t>::sumByLastAxis(
        DataContainer<std::complex<data_t>> dc) const
    {
        auto coeffsPerDim = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        index_t width = coeffsPerDim[0];
        index_t height = coeffsPerDim[1];
        index_t layers = coeffsPerDim[2];
        DataContainer<std::complex<data_t>> summedDC(VolumeDescriptor{{width, height}});

        for (index_t j = 0; j < width; j++) {
            for (index_t k = 0; k < height; k++) {
                std::complex<data_t> currValue = 0;
                for (index_t i = 0; i < layers; i++) {
                    currValue += dc(j, k, i);
                }
                summedDC(j, k) = currValue;
            }
        }

        return summedDC;
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::applyImpl(const DataContainer<ret_t>& x,
                                                     DataContainer<ret_t>& Ax) const
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
                   _width, _height, _numOfScales, _numOfLayers,
                   _isSpectraComputed ? "precomputed" : "non-precomputed");

        if (!isSpectraComputed()) {
            computeSpectra();
        }

        FourierTransform<std::complex<data_t>> fourierTransform(x.getDataDescriptor());

        DataContainer<std::complex<data_t>> fftImg = fourierTransform.apply(x.asComplex());

        for (index_t i = 0; i < getNumOfLayers(); i++) {
            DataContainer<std::complex<data_t>> temp =
                getSpectra().slice(i).viewAs(x.getDataDescriptor()).asComplex() * fftImg;
            if constexpr (isComplex<ret_t>) {
                Ax.slice(i) = fourierTransform.applyAdjoint(temp);
            } else {
                Ax.slice(i) = fourierTransform.applyAdjoint(temp).getReal();
            }
        }
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::applyAdjointImpl(const DataContainer<ret_t>& y,
                                                            DataContainer<ret_t>& Aty) const
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
                   _width, _height, _numOfScales, _numOfLayers,
                   _isSpectraComputed ? "precomputed" : "non-precomputed");

        if (!isSpectraComputed()) {
            computeSpectra();
        }

        FourierTransform<std::complex<data_t>> fourierTransform(Aty.getDataDescriptor());

        DataContainer<std::complex<data_t>> intermRes(y.getDataDescriptor());

        for (index_t i = 0; i < getNumOfLayers(); i++) {
            DataContainer<std::complex<data_t>> temp =
                fourierTransform.apply(y.slice(i).viewAs(Aty.getDataDescriptor()).asComplex())
                * getSpectra().slice(i).viewAs(Aty.getDataDescriptor()).asComplex();
            intermRes.slice(i) = fourierTransform.applyAdjoint(temp);
        }

        if constexpr (isComplex<ret_t>) {
            Aty = sumByLastAxis(intermRes);
        } else {
            Aty = sumByLastAxis(intermRes).getReal();
        }
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::computeSpectra() const
    {
        if (isSpectraComputed()) {
            Logger::get("ShearletTransform")->warn("Spectra have already been computed!");
        }

        DataContainer<data_t> spectra(VolumeDescriptor{{_width, _height, _numOfLayers}});
        spectra = 0;

        index_t i = 0;

        _computeSpectraAtLowFreq(spectra, i);

        for (index_t j = 0; j < _numOfScales; j++) {
            for (auto k = static_cast<index_t>(-std::pow(2, j));
                 k <= static_cast<index_t>(std::pow(2, j)); k++) {
                if (std::abs(k) <= static_cast<index_t>(std::pow(2, j)) - 1) {
                    _computeSpectraAtConicRegions(spectra, i, j, k);
                } else {
                    _computeSpectraAtSeamLines(spectra, i, j, k);
                }
            }
        }

        _spectra = spectra;

        _isSpectraComputed = true;
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtLowFreq(DataContainer<data_t>& spectra,
                                                                    index_t& i) const
    {
        DataContainer<data_t> sectionZero(VolumeDescriptor{{_width, _height}});
        sectionZero = 0;
        for (auto w = static_cast<int>(-std::floor(_width / 2.0)); w < std::ceil(_width / 2.0);
             w++) {
            for (auto h = static_cast<int>(-std::floor(_height / 2.0));
                 h < std::ceil(_height / 2.0); h++) {
                sectionZero(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                    phiHat(static_cast<data_t>(w), static_cast<data_t>(h));
            }
        }
        spectra.slice(i) = sectionZero;
        i += 1;
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtConicRegions(
        DataContainer<data_t>& spectra, index_t& i, index_t j, index_t k) const
    {
        DataContainer<data_t> sectionh(VolumeDescriptor{{_width, _height}});
        sectionh = 0;
        DataContainer<data_t> sectionv(VolumeDescriptor{{_width, _height}});
        sectionv = 0;
        for (auto w = static_cast<index_t>(-std::floor(_width / 2.0));
             w < static_cast<index_t>(std::ceil(_width / 2.0)); w++) {
            for (auto h = static_cast<index_t>(-std::floor(_height / 2.0));
                 h < static_cast<index_t>(std::ceil(_height / 2.0)); h++) {
                if (std::abs(h) <= std::abs(w)) {
                    sectionh(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        psiHat(std::pow(4, -j) * w, std::pow(4, -j) * k * w + std::pow(2, -j) * h);
                } else {
                    sectionv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        psiHat(std::pow(4, -j) * h, std::pow(4, -j) * k * h + std::pow(2, -j) * w);
                }
            }
        }
        spectra.slice(i) = sectionh;
        i += 1;
        spectra.slice(i) = sectionv;
        i += 1;
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtSeamLines(
        DataContainer<data_t>& spectra, index_t& i, index_t j, index_t k) const
    {
        DataContainer<data_t> sectionhxv(VolumeDescriptor{{_width, _height}});
        sectionhxv = 0;
        for (auto w = static_cast<index_t>(-std::floor(_width / 2.0));
             w < static_cast<index_t>(std::ceil(_width / 2.0)); w++) {
            for (auto h = static_cast<index_t>(-std::floor(_height / 2.0));
                 h < static_cast<index_t>(std::ceil(_height / 2.0)); h++) {
                if (std::abs(h) <= std::abs(w)) {
                    sectionhxv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        psiHat(std::pow(4, -j) * w, std::pow(4, -j) * k * w + std::pow(2, -j) * h);
                } else {
                    sectionhxv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        psiHat(std::pow(4, -j) * h, std::pow(4, -j) * k * h + std::pow(2, -j) * w);
                }
            }
        }
        spectra.slice(i) = sectionhxv;
        i += 1;
    }

    template <typename ret_t, typename data_t>
    data_t ShearletTransform<ret_t, data_t>::meyerFunction(data_t x) const
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

    template <typename ret_t, typename data_t>
    data_t ShearletTransform<ret_t, data_t>::b(data_t w) const
    {
        if (1 <= std::abs(w) && std::abs(w) <= 2) {
            return std::sin(pi<data_t> / 2.0 * meyerFunction(std::abs(w) - 1));
        } else if (2 < std::abs(w) && std::abs(w) <= 4) {
            return std::cos(pi<data_t> / 2.0 * meyerFunction(1.0 / 2 * std::abs(w) - 1));
        } else {
            return 0;
        }
    }

    template <typename ret_t, typename data_t>
    data_t ShearletTransform<ret_t, data_t>::phi(data_t w) const
    {
        if (std::abs(w) <= 1.0 / 2) {
            return 1;
        } else if (1.0 / 2 < std::abs(w) && std::abs(w) < 1) {
            return std::cos(pi<data_t> / 2.0 * meyerFunction(2 * std::abs(w) - 1));
        } else {
            return 0;
        }
    }

    template <typename ret_t, typename data_t>
    data_t ShearletTransform<ret_t, data_t>::phiHat(data_t w, data_t h) const
    {
        if (std::abs(h) <= std::abs(w)) {
            return phi(w);
        } else {
            return phi(h);
        }
    }

    template <typename ret_t, typename data_t>
    data_t ShearletTransform<ret_t, data_t>::psiHat1(data_t w) const
    {
        return std::sqrt(std::pow(b(2 * w), 2) + std::pow(b(w), 2));
    }

    template <typename ret_t, typename data_t>
    data_t ShearletTransform<ret_t, data_t>::psiHat2(data_t w) const
    {
        if (w <= 0) {
            return std::sqrt(meyerFunction(1 + w));
        } else {
            return std::sqrt(meyerFunction(1 - w));
        }
    }

    template <typename ret_t, typename data_t>
    data_t ShearletTransform<ret_t, data_t>::psiHat(data_t w, data_t h) const
    {
        if (w == 0) {
            return 0;
        } else {
            return psiHat1(w) * psiHat2(h / w);
        }
    }

    template <typename ret_t, typename data_t>
    auto ShearletTransform<ret_t, data_t>::getSpectra() const -> DataContainer<data_t>
    {
        if (!_spectra.has_value()) {
            throw LogicError(std::string("ShearletTransform: the spectra is not yet computed"));
        }
        return _spectra.value();
    }

    template <typename ret_t, typename data_t>
    bool ShearletTransform<ret_t, data_t>::isSpectraComputed() const
    {
        return _isSpectraComputed;
    }

    template <typename ret_t, typename data_t>
    index_t ShearletTransform<ret_t, data_t>::calculateNumOfScales(index_t width, index_t height)
    {
        return static_cast<index_t>(std::log2(std::max(width, height)) / 2.0);
    }

    template <typename ret_t, typename data_t>
    index_t ShearletTransform<ret_t, data_t>::calculateNumOfLayers(index_t width, index_t height)
    {
        return static_cast<index_t>(std::pow(2, (calculateNumOfScales(width, height) + 2)) - 3);
    }

    template <typename ret_t, typename data_t>
    index_t ShearletTransform<ret_t, data_t>::calculateNumOfLayers(index_t numOfScales)
    {
        return static_cast<index_t>(std::pow(2, numOfScales + 2) - 3);
    }

    template <typename ret_t, typename data_t>
    auto ShearletTransform<ret_t, data_t>::getWidth() const -> index_t
    {
        return _width;
    }

    template <typename ret_t, typename data_t>
    auto ShearletTransform<ret_t, data_t>::getHeight() const -> index_t
    {
        return _height;
    }

    template <typename ret_t, typename data_t>
    auto ShearletTransform<ret_t, data_t>::getNumOfLayers() const -> index_t
    {
        return _numOfLayers;
    }

    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>* ShearletTransform<ret_t, data_t>::cloneImpl() const
    {
        return new ShearletTransform<ret_t, data_t>(_width, _height, _numOfScales, _spectra);
    }

    template <typename ret_t, typename data_t>
    bool ShearletTransform<ret_t, data_t>::isEqual(const LinearOperator<ret_t>& other) const
    {
        if (!LinearOperator<ret_t>::isEqual(other))
            return false;

        auto otherST = downcast_safe<ShearletTransform<ret_t, data_t>>(&other);

        if (!otherST)
            return false;

        if (_width != otherST->_width)
            return false;

        if (_height != otherST->_height)
            return false;

        if (_numOfScales != otherST->_numOfScales)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ShearletTransform<float, float>;
    template class ShearletTransform<std::complex<float>, float>;
    template class ShearletTransform<double, double>;
    template class ShearletTransform<std::complex<double>, double>;
} // namespace elsa
