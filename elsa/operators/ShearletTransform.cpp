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
    DataContainer<elsa::complex<data_t>> ShearletTransform<ret_t, data_t>::sumByLastAxis(
        DataContainer<elsa::complex<data_t>> dc) const
    {
        auto coeffsPerDim = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        index_t width = coeffsPerDim[0];
        index_t height = coeffsPerDim[1];
        index_t layers = coeffsPerDim[2];
        DataContainer<elsa::complex<data_t>> summedDC(VolumeDescriptor{{width, height}});

        for (index_t j = 0; j < width; j++) {
            for (index_t k = 0; k < height; k++) {
                elsa::complex<data_t> currValue = 0;
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

        FourierTransform<elsa::complex<data_t>> fourierTransform(x.getDataDescriptor());

        DataContainer<elsa::complex<data_t>> fftImg = fourierTransform.apply(x.asComplex());

        for (index_t i = 0; i < getNumOfLayers(); i++) {
            DataContainer<elsa::complex<data_t>> temp =
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

        FourierTransform<elsa::complex<data_t>> fourierTransform(Aty.getDataDescriptor());

        DataContainer<elsa::complex<data_t>> intermRes(y.getDataDescriptor());

        for (index_t i = 0; i < getNumOfLayers(); i++) {
            DataContainer<elsa::complex<data_t>> temp =
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

        _computeSpectraAtLowFreq();

        for (index_t j = 0; j < _numOfScales; j++) {
            auto twoPowJ = static_cast<index_t>(std::pow(2, j));
            auto shearletsAtJ = static_cast<index_t>(std::pow(2, j + 2));
            index_t shearletsUpUntilJ = shearletsAtJ - 3;
            index_t index = 1;

            _computeSpectraAtSeamLines(j, -twoPowJ, shearletsUpUntilJ + twoPowJ);
            for (auto k = -twoPowJ + 1; k < twoPowJ; k++) {
                // modulo instead of remainder for negative numbers is needed here, therefore doing
                // "((a % b) + b) % b" instead of "a % b"
                index_t modIndex =
                    (((twoPowJ - index + 1) % shearletsAtJ) + shearletsAtJ) % shearletsAtJ;
                if (modIndex == 0) {
                    modIndex = shearletsAtJ - 1;
                } else {
                    --modIndex;
                }

                _computeSpectraAtConicRegions(j, k, shearletsUpUntilJ + modIndex,
                                              shearletsUpUntilJ + twoPowJ + index);
                ++index;
            }
            _computeSpectraAtSeamLines(j, twoPowJ, shearletsUpUntilJ + twoPowJ + index);
        }
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtLowFreq() const
    {
        DataContainer<data_t> sectionZero(VolumeDescriptor{{_width, _height}});
        sectionZero = 0;

        auto shape = getShapeFractions();

        // TODO attempt to refactor the negative indexing
        for (auto w = shape.negativeHalfWidth; w < shape.halfWidth; w++) {
            for (auto h = shape.negativeHalfHeight; h < shape.halfHeight; h++) {
                sectionZero(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                    shearlet::phiHat<data_t>(static_cast<data_t>(w), static_cast<data_t>(h));
            }
        }

        _spectra.value().slice(0) = sectionZero;
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtConicRegions(index_t j, index_t k,
                                                                         index_t hSliceIndex,
                                                                         index_t vSliceIndex) const
    {
        DataContainer<data_t> sectionh(VolumeDescriptor{{_width, _height}});
        sectionh = 0;
        DataContainer<data_t> sectionv(VolumeDescriptor{{_width, _height}});
        sectionv = 0;

        auto shape = getShapeFractions();
        auto jr = static_cast<data_t>(j);
        auto kr = static_cast<data_t>(k);

        // TODO attempt to refactor the negative indexing
        for (auto w = shape.negativeHalfWidth; w < shape.halfWidth; w++) {
            auto wr = static_cast<data_t>(w);
            for (auto h = shape.negativeHalfHeight; h < shape.halfHeight; h++) {
                auto hr = static_cast<data_t>(h);
                if (std::abs(h) <= std::abs(w)) {
                    sectionh(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        shearlet::psiHat<data_t>(std::pow(4.f, -jr) * wr,
                                                 std::pow(4.f, -jr) * kr * wr
                                                     + std::pow(2.f, -jr) * hr);
                } else {
                    sectionv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        shearlet::psiHat<data_t>(std::pow(4.f, -jr) * hr,
                                                 std::pow(4.f, -jr) * kr * hr
                                                     + std::pow(2.f, -jr) * wr);
                }
            }
        }

        _spectra.value().slice(hSliceIndex) = sectionh;
        _spectra.value().slice(vSliceIndex) = sectionv;
    }

    template <typename ret_t, typename data_t>
    void ShearletTransform<ret_t, data_t>::_computeSpectraAtSeamLines(index_t j, index_t k,
                                                                      index_t hxvSliceIndex) const
    {
        DataContainer<data_t> sectionhxv(VolumeDescriptor{{_width, _height}});
        sectionhxv = 0;

        auto shape = getShapeFractions();
        auto jr = static_cast<data_t>(j);
        auto kr = static_cast<data_t>(k);

        // TODO attempt to refactor the negative indexing
        for (auto w = shape.negativeHalfWidth; w < shape.halfWidth; w++) {
            auto wr = static_cast<data_t>(w);
            for (auto h = shape.negativeHalfHeight; h < shape.halfHeight; h++) {
                auto hr = static_cast<data_t>(h);
                if (std::abs(h) <= std::abs(w)) {
                    sectionhxv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        shearlet::psiHat<data_t>(std::pow(4.f, -jr) * wr,
                                                 std::pow(4.f, -jr) * kr * wr
                                                     + std::pow(2.f, -jr) * hr);
                } else {
                    sectionhxv(w < 0 ? w + _width : w, h < 0 ? h + _height : h) =
                        shearlet::psiHat<data_t>(std::pow(4.f, -jr) * hr,
                                                 std::pow(4.f, -jr) * kr * hr
                                                     + std::pow(2.f, -jr) * wr);
                }
            }
        }

        _spectra.value().slice(hxvSliceIndex) = sectionhxv;
    }

    /**
     * helper function to calculate input data fractions.
     */
    template <typename ret_t, typename data_t>
    auto ShearletTransform<ret_t, data_t>::getShapeFractions() const -> shape_fractions
    {
        shape_fractions ret;
        auto width = static_cast<real_t>(_width);
        auto height = static_cast<real_t>(_height);

        ret.negativeHalfWidth = static_cast<index_t>(-std::floor(width / 2.0));
        ret.halfWidth = static_cast<index_t>(std::ceil(width / 2.0));
        ret.negativeHalfHeight = static_cast<index_t>(-std::floor(height / 2.0));
        ret.halfHeight = static_cast<index_t>(std::ceil(height / 2.0));

        return ret;
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
    template class ShearletTransform<elsa::complex<float>, float>;
    template class ShearletTransform<double, double>;
    template class ShearletTransform<elsa::complex<double>, double>;
} // namespace elsa
