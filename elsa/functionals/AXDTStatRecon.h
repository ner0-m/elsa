#pragma once

#include "Functional.h"
#include "RandomBlocksDescriptor.h"
#include "VolumeDescriptor.h"

namespace elsa
{
    /**
     * @brief Class representing the AXDT Statistical Reconstruction functional.
     *
     * @author Shen Hu - initial code
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * There are four reconstruction modes implemented.
     * Given the formula: S_n = a + b * cos(2*pi*n/N - phi),
     * and let d be the dark-field signal:
     * 1) Mode Gaussian_log_d: assume log(d) follows a Gaussian distribution
     * 2) Mode Gaussian_d: assume d follows a Gaussian distribution
     * 3) Mode Gaussian_approximate_racian: assume a follows Gaussian distribution,
     * and b follows a Racian distribution, but use a Gaussian distribution to approximate
     * the random variable b for a simpler pdf
     * 4) Mode Racian_direct: similar to 3), but model the Racian distribution directly
     *
     * see http://onlinelibrary.fully3d.org/papers/2017/Fully3D.2017-11-3203019.pdf for details,
     * (this implementation follows the notation in the paper)
     *
     * Also, this functional(AXDTStatRecon) does not wrap any residuals
     */
    template <typename data_t = real_t>
    class AXDTStatRecon : public Functional<data_t>
    {
    public:
        /// enum representing the four aforementioned statistical models
        enum StatReconType {
            Gaussian_log_d,
            Gaussian_d,
            Gaussian_approximate_racian,
            Racian_direct
        };

        /**
         * @brief full constructor for the AXDTStatRecon functional,
         * mapping domain vector to a scalar (without a residual)
         *
         * @param[in] ffa the flat-field measured value of RV a
         * @param[in] ffb the flat-field measured value of RV b
         * @param[in] a the measured value of RV a (with the measured sample in-place)
         * @param[in] b the measured value of RV b (with the measured sample in-place)
         * @param[in] absorp_op operator modeling the projection of the regular absorption signal
         * @param[in] axdt_op operator modeling the projection of the dark-field signal
         * @param[in] N total grating-stepping steps
         * @param[in] recon_type enum specifying the statistical model for reconstruction
         */
        AXDTStatRecon(const DataContainer<data_t>& ffa,
                      const DataContainer<data_t>& ffb,
                      const DataContainer<data_t>& a,
                      const DataContainer<data_t>& b,
                      const LinearOperator<data_t>& absorp_op,
                      const LinearOperator<data_t>& axdt_op,
                      index_t N,
                      const StatReconType& recon_type
                      );

        /**
         * @brief short-hand constructor for the AXDTStatRecon functional,
         * available only for Mode Gaussian_log_d and Gaussian_d
         *
         * @param[in] axdt_proj the measured value of -log(d) (with the measured sample in-place)
         * @param[in] axdt_op operator modeling the projection of the dark-field signal
         * @param[in] recon_type enum specifying the statistical model for reconstruction
         *
         * @throw InvalidArgumentError if mode other than Gaussian_log_d or Gaussian_d is
         * specified in this constructor
         */
        AXDTStatRecon(const DataContainer<data_t>& axdt_proj,
                      const LinearOperator<data_t>& axdt_op,
                      const StatReconType& recon_type
        );

        /// make copy constructor deletion explicit
        AXDTStatRecon(const AXDTStatRecon<data_t>&) = delete;

        /// default destructor
        ~AXDTStatRecon() override = default;

    protected:
        /// the evaluation of this functional
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        AXDTStatRecon<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

        /// generate a minimal volume descriptor as a placeholder when absorption data missing
        static std::unique_ptr<DataDescriptor> generate_placeholder_descriptor();

        /// construct a RandomBlockDescriptor to wrap up the absorption projection data
        /// and the axdt projection data as a single input for this functional
        static RandomBlocksDescriptor generate_descriptors(const DataDescriptor& desc1,
                                                     const DataDescriptor& desc2);

    private:
        /// the flat-field measured value of RV a
        std::optional<DataContainer<data_t>> ffa_;

        ///the flat-field measured value of RV b
        std::optional<DataContainer<data_t>> ffb_;

        /// the measured value of RV a
        std::optional<DataContainer<data_t>> a_tilde_;

        /// the measured value of RV b
        std::optional<DataContainer<data_t>> b_tilde_;

        /// operator modeling the projection of the regular absorption signal
        std::unique_ptr<LinearOperator<data_t>> absorp_op_;

        /// operator modeling the projection of the dark-field signal
        std::unique_ptr<LinearOperator<data_t>> axdt_op_;

        /// store the frequently used value alpha (= ffb / ffa)
        std::optional<DataContainer<data_t>> alpha_;

        /// store the frequently used value d_tilde (= b_tilde / a_tilde / alpha)
        std::optional<DataContainer<data_t>> d_tilde_;

        /// total grating-stepping steps
        data_t N_;

        /// intended statistical model for reconstruction
        StatReconType recon_type_;
    };

    namespace axdt {
        /// element-wise logarithm of modified Bessel function of the first
        /// kind of order zero. i.e. log(B_i(0, x))
        template <typename data_t>
        DataContainer<data_t> log_bessel_0(const DataContainer<data_t>& x) {
            return bessel_log_0(x);
        }

        /// element-wise quotient between modified Bessel function of the first
        /// kind of order one and that of order zero. i.e. B_i(1, x) / B_i(0, x)
        template <typename data_t>
        DataContainer<data_t> quot_bessel_1_0(const DataContainer<data_t>& x) {
            return bessel_1_0(x);
        }

    }  // namespace axdt

} // namespace elsa
