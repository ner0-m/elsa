#pragma once

#include "DataDescriptor.h"

namespace elsa
{

    /**
     * \brief Class representing metadata for linearized n-dimensional signal stored in memory
     *
     * \author Matthias Wieczorek - initial code
     * \author Tobias Lasser - modularization, modernization
     * \author Maximilian Hornung - various enhancements
     * \author David Frank - inheritance restructuring
     *
     * This class provides metadata about a signal that is stored in memory (typically a
     * DataContainer). This signal can be n-dimensional, and will be stored in memory in a
     * linearized fashion.
     */
    class VolumeDescriptor : public DataDescriptor
    {
    public:
        /// delete default constructor (having no metadata is invalid)
        VolumeDescriptor() = delete;

        /// default destructor
        ~VolumeDescriptor() override = default;

        /**
         * \brief Constructor for DataDescriptor, accepts vector for coefficients per dimensions
         *
         * \param[in] numberOfCoefficientsPerDimension vector containing the number of coefficients
         * per dimension, (dimension is set implicitly from the size of the vector)
         *
         * \throw InvalidArgumentError if any number of coefficients is non-positive
         */
        explicit VolumeDescriptor(IndexVector_t numberOfCoefficientsPerDimension);

        /**
         * \brief Constructs VolumeDescriptor from initializer list for the coefficients per
         * dimensions
         *
         * \param[in] numberOfCoefficientsPerDimension initializer list containing the number of
         * coefficients per dimension (dimension is set implicitly from the size of the list)
         *
         * \throw InvalidArgumentError if any number of coefficients is non-positive
         */
        explicit VolumeDescriptor(std::initializer_list<index_t> numberOfCoefficientsPerDimension);

        /**
         * \brief Constructor for DataDescriptor, accepts vectors for coefficients and spacing
         *
         * \param[in] numberOfCoefficientsPerDimension vector containing the number of coefficients
         * per dimension, (dimension is set implicitly from the size of the vector)
         * \param[in] spacingPerDimension vector containing the spacing per dimension
         *
         * \throw InvalidArgumentError if any number of coefficients or spacing is non-positive,
         *        or sizes of numberOfCoefficientsPerDimension and spacingPerDimension do not match
         */
        explicit VolumeDescriptor(IndexVector_t numberOfCoefficientsPerDimension,
                                  RealVector_t spacingPerDimension);

        /**
         * \brief Constructs VolumeDescriptor from two initializer lists for coefficients and
         * spacing
         *
         * \param[in] numberOfCoefficientsPerDimension initializer list containing the number of
         * coefficients per dimension (dimension is set implicitly from the size of the list)
         * \param[in] spacingPerDimension initializer list containing the spacing per dimension
         *
         * \throw InvalidArgumentError if any number of coefficients or spacing is non-positive,
         *        or sizes of numberOfCoefficientsPerDimension and spacingPerDimension do not match
         */
        explicit VolumeDescriptor(std::initializer_list<index_t> numberOfCoefficientsPerDimension,
                                  std::initializer_list<real_t> spacingPerDimension);

    private:
        /// implement the polymorphic clone operation
        VolumeDescriptor* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataDescriptor& other) const override;
    };

} // namespace elsa
