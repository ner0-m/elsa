#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"

#include <vector>

namespace elsa
{

    /**
     * @brief Base class for representing metadata for linearized n-dimensional signal stored in
     * memory
     *
     * @author Matthias Wieczorek - initial code
     * @author Tobias Lasser - modularization, modernization
     * @author Maximilian Hornung - various enhancements
     * @author David Frank - inheritance restructuring
     *
     * This class provides an interface for metadata about a signal that is stored in memory. This
     * base class provides other descriptor subclasses with a fundamental interface to access the
     * important parameters (i.e. dimensionality, size and spacing) of the metadata.
     *
     */
    class DataDescriptor : public Cloneable<DataDescriptor>
    {
    public:
        /// delete default constructor (having no metadata is invalid)
        DataDescriptor() = delete;

        /// Pure virtual destructor
        virtual ~DataDescriptor() = 0;

        /**
         * @brief Constructor for DataDescriptor, accepts dimension and size
         *
         * @param[in] numberOfCoefficientsPerDimension vector containing the number of coefficients
         * per dimension, (dimension is set implicitly from the size of the vector)
         *
         * @throw InvalidArgumentError if any number of coefficients is non-positive
         */
        explicit DataDescriptor(IndexVector_t numberOfCoefficientsPerDimension);

        /**
         * @brief Constructor for DataDescriptor, accepts dimension, size and spacing
         *
         * @param[in] numberOfCoefficientsPerDimension vector containing the number of coefficients
         * per dimension, (dimension is set implicitly from the size of the vector)
         * @param[in] spacingPerDimension vector containing the spacing per dimension
         *
         * @throw InvalidArgumentError if any number of coefficients is non-positive,
         *        or sizes of numberOfCoefficientsPerDimension and spacingPerDimension do not match
         */
        explicit DataDescriptor(IndexVector_t numberOfCoefficientsPerDimension,
                                RealVector_t spacingPerDimension);

        /// return the number of dimensions
        index_t getNumberOfDimensions() const;

        /// return the total number of coefficients
        index_t getNumberOfCoefficients() const;

        /// return the number of coefficients per dimension
        IndexVector_t getNumberOfCoefficientsPerDimension() const;

        /// return the product of coefficients per dimension
        IndexVector_t getProductOfCoefficientsPerDimension() const;

        /// return the spacing per dimension
        RealVector_t getSpacingPerDimension() const;

        /// return the location of the origin (typically the center)
        RealVector_t getLocationOfOrigin() const;

        /**
         * @brief computes the linearized index in the data vector from local coordinates
         *
         * @param[in] coordinate vector containing the local coordinate
         * @return the index into the linearized data vector
         *
         * The local coordinates are integers, running from 0 to
         * _numberOfCoefficientsPerDimension[i]-1 for every dimension i = 0,...,_numberOfDimensions.
         * Linearization is assumed to be done in order of the dimensions.
         */
        index_t getIndexFromCoordinate(const elsa::IndexVector_t& coordinate) const;

        /**
         * @brief computes the local coordinates from a linearized index
         *
         * @param[in] index into the linearized data vector
         * @return the local coordinate corresponding to the index
         *
         * The local coordinates are integers, running from 0 to
         * _numberOfCoefficientsPerDimension[i]-1 for every dimension i =
         * 0,...,_numberOfDimensions. Linearization is assumed to be done in order of the
         * dimensions.
         */
        IndexVector_t getCoordinateFromIndex(index_t index) const;

        template <index_t dim>
        index_t coord2Idx(Eigen::Matrix<real_t, dim, 1> coord) const
        {
            return _productOfCoefficientsPerDimension.cwiseProduct(coord).sum();
        }

        template <index_t dim>
        Eigen::Matrix<index_t, dim, 1> idx2Coord(index_t idx) const
        {
            Eigen::Matrix<index_t, dim, 1> coordinate;

            index_t leftOver = idx;
            for (index_t i = dim - 1; i >= 1; --i) {
                coordinate[i] = leftOver / _productOfCoefficientsPerDimension[i];
                leftOver %= _productOfCoefficientsPerDimension[i];
            }
            coordinate[0] = leftOver;

            return coordinate;
        }

    protected:
        /// Number of dimensions
        index_t _numberOfDimensions;

        /// vector containing the number of coefficients per dimension
        IndexVector_t _numberOfCoefficientsPerDimension;

        /// vector containing the spacing per dimension
        RealVector_t _spacingPerDimension;

        /// vector containing precomputed partial products of coefficients per dimension for index
        /// computations
        IndexVector_t _productOfCoefficientsPerDimension;

        /// vector containing the origin of the described volume (typically the center)
        RealVector_t _locationOfOrigin;

        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        DataDescriptor(const DataDescriptor&) = default;

        /// default move constructor, hidden from non-derived classes to prevent potential slicing
        DataDescriptor(DataDescriptor&&) = default;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataDescriptor& other) const override;
    };

} // namespace elsa
