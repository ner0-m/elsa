#pragma once

#include "LinearOperator.h"

namespace elsa
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
     * computation is taken for the spectra, which is stored after the first run. It only handles
     * signals with one channel, e.g. grayscale images. Increasing the number of scales will
     * increase precision.
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

        /// method for computing the spectra, should only be called once as subsequent calls will
        /// generate the same spectra
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
        //  manner, but that might take quite some time, can this make it to master in the meantime?
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

        /// variable indicating if the spectra is already computed
        mutable bool _isSpectraComputed{false};

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

        /// defined in Y. Meyer. Oscillating Patterns in Image Processing and Nonlinear Evolution
        /// Equations. AMS, 2001.
        data_t meyerFunction(data_t x) const;

        data_t b(data_t w) const;

        data_t phi(data_t w) const;

        data_t phiHat(data_t w, data_t h) const;

        data_t psiHat1(data_t w) const;

        data_t psiHat2(data_t w) const;

        data_t psiHat(data_t w, data_t h) const;

        void _computeSpectraAtLowFreq(DataContainer<data_t>& spectra, index_t& i) const;

        void _computeSpectraAtConicRegions(DataContainer<data_t>& spectra, index_t& i, index_t j,
                                           index_t k) const;

        void _computeSpectraAtSeamLines(DataContainer<data_t>& spectra, index_t& i, index_t j,
                                        index_t k) const;
    };
} // namespace elsa
