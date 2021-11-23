#include "ShearletTransform.h"
#include "FourierTransform.h"
#include "VolumeDescriptor.h"
#include "Timer.h"
#include "Math.hpp"

namespace elsa
{
    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>::ShearletTransform(IndexVector_t spatialDimensions)
        : ShearletTransform(spatialDimensions[0], spatialDimensions[1])
    {
        if (spatialDimensions.size() != 2) {
            throw LogicError("ShearletTransform: Only 2D shape supported");
        }
    }

    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>::ShearletTransform(index_t width, index_t height)
        : ShearletTransform(width, height, calculateNumOfScales(width, height))
    {
    }

    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>::ShearletTransform(index_t width, index_t height,
                                                        index_t numOfScales)
        : ShearletTransform(width, height, numOfScales, std::nullopt)
    {
    }

    template <typename ret_t, typename data_t>
    ShearletTransform<ret_t, data_t>::ShearletTransform(
        index_t width, index_t height, index_t numOfScales,
        std::optional<DataContainer<data_t>> spectra)
        : LinearOperator<ret_t>(
            VolumeDescriptor{{width, height}},
            VolumeDescriptor{{width, height, calculateNumOfLayers(numOfScales)}}),
          _spectra{spectra},
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
                   isSpectraComputed() ? "precomputed" : "non-precomputed");

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
                Ax.slice(i) = real(fourierTransform.applyAdjoint(temp));
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
            ->info("Running the inverse shearlet transform on a 2D signal of shape ({}, {}), "
                   "on {} "
                   "scales with an oversampling factor of {} and {} spectra",
                   _width, _height, _numOfScales, _numOfLayers,
                   isSpectraComputed() ? "precomputed" : "non-precomputed");

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
            Aty = real(sumByLastAxis(intermRes));
        }
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::computeSpectra() const
    {
        if (isSpectraComputed()) {
            Logger::get("ShearletTransform")->warn("Spectra have already been computed!");
        }

        _spectra = DataContainer<data_t>(VolumeDescriptor{{_width, _height, _numOfLayers}});

        index_t i = 0;

        _computeSpectraAtLowFreq(i);

        for (index_t j = 0; j < _numOfScales; j++) {
            auto twoPowJ = static_cast<index_t>(std::pow(2, j));

            _computeSpectraAtSeamLines(i, j, -twoPowJ);
            for (auto k = -twoPowJ + 1; k < twoPowJ; k++) {
                _computeSpectraAtConicRegions(i, j, k);
            }
            _computeSpectraAtSeamLines(i, j, twoPowJ);
        }

        assert(i == _numOfLayers
               && "ShearletTransform: The layers of the spectra were indexed wrong");
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtLowFreq(index_t& i) const
    {
        DataContainer<data_t> sectionZero(VolumeDescriptor{{_width, _height}});
        sectionZero = 0;

        auto negativeHalfWidth = static_cast<index_t>(-std::floor(_width / 2.0));
        auto halfWidth = static_cast<index_t>(std::ceil(_width / 2.0));
        auto negativeHalfHeight = static_cast<index_t>(-std::floor(_height / 2.0));
        auto halfHeight = static_cast<index_t>(std::ceil(_height / 2.0));

        // TODO attempt to refactor the negative indexing
        for (auto w = negativeHalfWidth; w < halfWidth; w++) {
            for (auto h = negativeHalfHeight; h < halfHeight; h++) {
                sectionZero(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                    shearlet::phiHat<data_t>(static_cast<data_t>(w), static_cast<data_t>(h));
            }
        }

        _spectra.value().slice(i++) = sectionZero;
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtConicRegions(index_t& i, index_t j,
                                                                         index_t k) const
    {
        DataContainer<data_t> sectionh(VolumeDescriptor{{_width, _height}});
        sectionh = 0;
        DataContainer<data_t> sectionv(VolumeDescriptor{{_width, _height}});
        sectionv = 0;

        auto negativeHalfWidth = static_cast<index_t>(-std::floor(_width / 2.0));
        auto halfWidth = static_cast<index_t>(std::ceil(_width / 2.0));
        auto negativeHalfHeight = static_cast<index_t>(-std::floor(_height / 2.0));
        auto halfHeight = static_cast<index_t>(std::ceil(_height / 2.0));

        // TODO attempt to refactor the negative indexing
        for (auto w = negativeHalfWidth; w < halfWidth; w++) {
            for (auto h = negativeHalfHeight; h < halfHeight; h++) {
                if (std::abs(h) <= std::abs(w)) {
                    sectionh(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        shearlet::psiHat<data_t>(std::pow(4, -j) * w,
                                                 std::pow(4, -j) * k * w + std::pow(2, -j) * h);
                } else {
                    sectionv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        shearlet::psiHat<data_t>(std::pow(4, -j) * h,
                                                 std::pow(4, -j) * k * h + std::pow(2, -j) * w);
                }
            }
        }

        _spectra.value().slice(i++) = sectionh;
        _spectra.value().slice(i++) = sectionv;
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtSeamLines(index_t& i, index_t j,
                                                                      index_t k) const
    {
        DataContainer<data_t> sectionhxv(VolumeDescriptor{{_width, _height}});
        sectionhxv = 0;

        auto negativeHalfWidth = static_cast<index_t>(-std::floor(_width / 2.0));
        auto halfWidth = static_cast<index_t>(std::ceil(_width / 2.0));
        auto negativeHalfHeight = static_cast<index_t>(-std::floor(_height / 2.0));
        auto halfHeight = static_cast<index_t>(std::ceil(_height / 2.0));

        // TODO attempt to refactor the negative indexing
        for (auto w = negativeHalfWidth; w < halfWidth; w++) {
            for (auto h = negativeHalfHeight; h < halfHeight; h++) {
                if (std::abs(h) <= std::abs(w)) {
                    sectionhxv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        shearlet::psiHat<data_t>(std::pow(4, -j) * w,
                                                 std::pow(4, -j) * k * w + std::pow(2, -j) * h);
                } else {
                    sectionhxv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        shearlet::psiHat<data_t>(std::pow(4, -j) * h,
                                                 std::pow(4, -j) * k * h + std::pow(2, -j) * w);
                }
            }
        }

        _spectra.value().slice(i++) = sectionhxv;
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
        return _spectra.has_value();
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
