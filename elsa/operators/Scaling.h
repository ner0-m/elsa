#pragma once

#include "LinearOperator.h"

namespace elsa
{
    /**
     * \brief Operator representing a scaling operation.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - minor fixes
     * \author Tobias Lasser - modularization, rewrite
     *
     * \tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * This class represents a linear operator A that scales the input, either by a scalar
     * or by a diagonal scaling matrix.
     */
    template <typename data_t = real_t>
    class Scaling : public LinearOperator<data_t>
    {
    public:
        /**
         * \brief Constructor for a scalar, isotropic scaling operator.
         *
         * \param[in] descriptor DataDescriptor describing the domain and the range of the operator
         * \param[in] scaleFactor the scalar factor to scale with
         */
        Scaling(const DataDescriptor& descriptor, data_t scaleFactor);

        /**
         * \brief Constructor for a diagonal, anisotropic scaling operator.
         *
         * \param[in] descriptor DataDescriptor describing the domain and the range of the operator
         * \param[in] scaleFactors a DataContainer containing the scaling factor to be put on the
         * diagonal
         */
        Scaling(const DataDescriptor& descriptor, const DataContainer<data_t>& scaleFactors);

        /// default destructor
        ~Scaling() override = default;

        /// is the scaling isotropic
        bool isIsotropic() const;

        /// returns the scale factor (throws if scaling is not isotropic)
        data_t getScaleFactor() const;

        /// returns the scale factors (throws if scaling is isotropic)
        const DataContainer<data_t>& getScaleFactors() const;

    protected:
        /// apply the scaling operation
        void _apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the scaling operation
        void _applyAdjoint(const DataContainer<data_t>& y,
                           DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        Scaling<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// flag if the scaling is isotropic
        bool _isIsotropic;

        /// isotropic scaling factor
        data_t _scaleFactor;

        /// anisotropic scaling factors
        std::unique_ptr<DataContainer<data_t>> _scaleFactors{};
    };

} // namespace elsa
