#pragma once

#include "LinearOperator.h"
#include "Math.hpp"

namespace elsa::shearlets
{
    /**
     * @brief Class representing a (regular) Cone-Adapted Discrete Shearlet Transform
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * ShearletTransform represents a band-limited (compact support in Fourier domain)
     * representation system. It oversamples a 2D signal of (W, H) to (W, H, L). Most of the
     * computation is taken for the spectra, which is stored after the first run. It only
     * handles signals with one channel, e.g. grayscale images. Increasing the number of scales
     * will increase precision.
     *
     * References:
     * https://www.math.uh.edu/~dlabate/SHBookIntro.pdf
     * https://www.math.uh.edu/~dlabate/Athens.pdf
     * https://arxiv.org/pdf/1202.1773.pdf
     */
    template <typename ret_t = real_t, typename data_t = real_t>
    class ShearletTransform : public LinearOperator<ret_t>
    {
    public:
        /**
         * @brief Constructor for a (regular) cone-adapted discrete shearlet transform.
         *
         * @param[in] spatialDimensions the width and height of the input image
         */
        ShearletTransform(IndexVector_t spatialDimensions);

        /**
         * @brief Constructor for a (regular) cone-adapted discrete shearlet transform.
         *
         * @param[in] width the width of the input image
         * @param[in] height the height of the input image
         */
        ShearletTransform(index_t width, index_t height);

        /**
         * @brief Constructor for a (regular) cone-adapted discrete shearlet transform.
         *
         * @param[in] width the width of the input image
         * @param[in] height the height of the input image
         * @param[in] numOfScales the number of scales
         */
        ShearletTransform(index_t width, index_t height, index_t numOfScales);

        /**
         * @brief Constructor for a (regular) cone-adapted discrete shearlet transform.
         *
         * @param[in] width the width of the input image
         * @param[in] height the height of the input image
         * @param[in] numOfScales the number of scales
         * @param[in] spectra the spectra
         */
        ShearletTransform(index_t width, index_t height, index_t numOfScales,
                          std::optional<DataContainer<data_t>> spectra);

        /// default destructor
        ~ShearletTransform() override = default;

        /// method for computing the spectra, should only be called once as subsequent calls
        /// will generate the same spectra
        void computeSpectra() const;

        /// method indicating if the spectra has already been computed
        bool isSpectraComputed() const;

        /// return the spectra
        auto getSpectra() const -> DataContainer<data_t>;

        /// return the width
        auto getWidth() const -> index_t;

        /// return the height
        auto getHeight() const -> index_t;

        /// return the oversampling factor
        auto getNumOfLayers() const -> index_t;

        // TODO ideally this ought to be implemented somewhere else, perhaps in a more general
        //  manner, but that might take quite some time, can this make it to master in the
        //  meantime?
        DataContainer<std::complex<data_t>>
            sumByLastAxis(DataContainer<std::complex<data_t>> container) const;

    protected:
        void applyImpl(const DataContainer<ret_t>& x, DataContainer<ret_t>& Ax) const override;

        void applyAdjointImpl(const DataContainer<ret_t>& y,
                              DataContainer<ret_t>& Aty) const override;

        /// implement the polymorphic clone operation
        ShearletTransform<ret_t, data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<ret_t>& other) const override;

    private:
        /// variable to store the spectra
        mutable std::optional<DataContainer<data_t>> _spectra = std::nullopt;

        /// variables to store the spatial extents
        index_t _width;
        index_t _height;

        /// variable to store the number of scales
        index_t _numOfScales;

        index_t calculateNumOfScales(index_t width, index_t height);

        /// variable to store the oversampling factor
        index_t _numOfLayers;

        index_t calculateNumOfLayers(index_t width, index_t height);

        index_t calculateNumOfLayers(index_t numOfScales);

        void _computeSpectraAtLowFreq(index_t& i) const;

        void _computeSpectraAtConicRegions(index_t& i, index_t j, index_t k) const;

        void _computeSpectraAtSeamLines(index_t& i, index_t j, index_t k) const;

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        data_t b(data_t w) const
        {
            if (1 <= std::abs(w) && std::abs(w) <= 2) {
                return std::sin(pi<data_t> / 2.0 * meyerFunction(std::abs(w) - 1));
            } else if (2 < std::abs(w) && std::abs(w) <= 4) {
                return std::cos(pi<data_t> / 2.0 * meyerFunction(1.0 / 2 * std::abs(w) - 1));
            } else {
                return 0;
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        data_t phi(data_t w) const
        {
            if (std::abs(w) <= 1.0 / 2) {
                return 1;
            } else if (1.0 / 2 < std::abs(w) && std::abs(w) < 1) {
                return std::cos(pi<data_t> / 2.0 * meyerFunction(2 * std::abs(w) - 1));
            } else {
                return 0;
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        data_t phiHat(data_t w, data_t h) const
        {
            if (std::abs(h) <= std::abs(w)) {
                return phi(w);
            } else {
                return phi(h);
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        data_t psiHat1(data_t w) const
        {
            return std::sqrt(std::pow(b(2 * w), 2) + std::pow(b(w), 2));
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        data_t psiHat2(data_t w) const
        {
            if (w <= 0) {
                return std::sqrt(meyerFunction(1 + w));
            } else {
                return std::sqrt(meyerFunction(1 - w));
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        data_t psiHat(data_t w, data_t h) const
        {
            if (w == 0) {
                return 0;
            } else {
                return psiHat1(w) * psiHat2(h / w);
            }
        }
    };
} // namespace elsa::shearlets
