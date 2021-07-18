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
     * signals with one channel, e.g. grayscale images. Increasing the number of scales jZero will
     * increase precision.
     *
     * References:
     * https://www.math.uh.edu/~dlabate/SHBookIntro.pdf
     * https://www.math.uh.edu/~dlabate/Athens.pdf
     * https://arxiv.org/pdf/1202.1773.pdf
     */
    template <typename data_t = real_t>
    class ShearletTransform : public LinearOperator<data_t>
    {
    public:
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
         * @param[in] jZero the number of scales
         */
        ShearletTransform(index_t width, index_t height, index_t jZero);

        //        /**
        //         * @brief Constructor for a (regular) cone-adapted discrete shearlet transform.
        //         *
        //         * @param[in] width the width of the input image
        //         * @param[in] height the height of the input image
        //         * @param[in] jZero the number of scales for
        //         */
        //        ShearletTransform(index_t width, index_t height, index_t jZero, ShearletType:
        //        smooth etc);

        /// default destructor
        ~ShearletTransform() override = default;

        /// method for computing the spectra, should only be called once as subsequent calls will
        /// generate the same spectra
        void computeSpectra() const;

        /// method indicating if the spectra has already been computed
        bool isSpectraComputed() const;

        /// return the spectra
        auto getSpectra() const -> DataContainer<data_t>;

    protected:
        void applyImpl(const DataContainer<data_t>& f, DataContainer<data_t>& SHf) const override;

        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& SHty) const override;

        /// implement the polymorphic clone operation
        ShearletTransform<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// variable to store the spectra
        mutable std::unique_ptr<DataContainer<data_t>> _spectra;

        /// variable indicating if the spectra is already computed
        mutable bool _isSpectraComputed{false};

        /// variables to store the spatial extents
        index_t _width{-1};
        index_t _height{-1};

        /// variable to store the number of scales
        index_t _jZero{-1};

        index_t calculatejZero(index_t width, index_t height);

        /// variable to store the oversampling factor
        index_t _L{-1};

        index_t calculateL(index_t width, index_t height);

        /// defined in Y. Meyer. Oscillating Patterns in Image Processing and Nonlinear Evolution
        /// Equations. AMS, 2001.
        data_t meyerFunction(data_t x) const;

        data_t b(data_t w) const;

        data_t phi(data_t w) const;

        data_t phiHat(data_t w1, data_t w2) const;

        data_t psiHat1(data_t w) const;

        data_t psiHat2(data_t w) const;

        data_t psiHat(data_t w1, data_t w2) const;
    };
} // namespace elsa
