#pragma once


#include "elsa.h"
#include "DataHandler.h"

#include <Eigen/Core>

namespace elsa
{

    /**
     * \brief Class representing and owning a vector stored in CPU main memory (using Eigen::Matrix).
     *
     * \tparam data_t - data type that is stored, defaulting to real_t.
     *
     * \author David Frank - main code
     * \author Tobias Lasser - modularization and modernization
     */
     template <typename data_t = real_t>
     class DataHandlerCPU : public DataHandler<data_t>
     {
     protected:
         /// convenience typedef for the Eigen::Matrix data vector
         using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

     public:
         /// delete default constructor (having no information makes no sense)
         DataHandlerCPU() = delete;

         /// default destructor
         ~DataHandlerCPU() override = default;

         /**
          * \brief Constructor initializing an appropriately sized vector with zeros
          *
          * \param[in] size of the vector
          * \param[in] initialize - set to false if you do not need initialization with zeros (default: true)
          *
          * \throw std::invalid_argument if the size is non-positive
          */
         explicit DataHandlerCPU(index_t size, bool initialize = true);

         /**
          * \brief Constructor initializing a data vector with a given vector
          *
          * \param[in] vector that is used for initializing the data
          */
         explicit DataHandlerCPU(DataVector_t vector);


         /// return the size of the vector
        index_t getSize() const override;

         /// return the index-th element of the data vector (not bounds checked!)
         data_t& operator[](index_t index) override;

         /// return the index-th element of the data vector as read-only (not bound checked!)
         const data_t& operator[](index_t index) const override;


         /// return the dot product of the data vector with vector v
         data_t dot(const DataHandler<data_t>& v) const override;

         /// return the squared l2 norm of the data vector (dot product with itself)
         data_t squaredNorm() const override;

         /// return the sum of all elements of the data vector
         data_t sum() const override;


         /// return a new DataHandler with element-wise squared values of this one
         std::unique_ptr<DataHandler<data_t>> square() const override;

         /// return a new DataHandler with element-wise square roots of this one
         std::unique_ptr<DataHandler<data_t>> sqrt() const override;

         /// return a new DataHandler with element-wise exponentials of this one
         std::unique_ptr<DataHandler<data_t>> exp() const override;

         /// return a new DataHandler with element-wise logarithms of this one
         std::unique_ptr<DataHandler<data_t>> log() const override;


         /// compute in-place element-wise addition of another vector v
         DataHandler<data_t>& operator+=(const DataHandler<data_t>& v) override;

         /// compute in-place element-wise subtraction of another vector v
         DataHandler<data_t>& operator-=(const DataHandler<data_t>& v) override;

         /// compute in-place element-wise multiplication by another vector v
         DataHandler<data_t>& operator*=(const DataHandler<data_t>& v) override;

         /// compute in-place element-wise division by another vector v
         DataHandler<data_t>& operator/=(const DataHandler<data_t>& v) override;


         /// compute in-place addition of a scalar
         DataHandler<data_t>& operator+=(data_t scalar) override;

         /// compute in-place subtraction of a scalar
         DataHandler<data_t>& operator-=(data_t scalar) override;

         /// compute in-place multiplication by a scalar
         DataHandler<data_t>& operator*=(data_t scalar) override;

         /// compute in-place division by a scalar
         DataHandler<data_t>& operator/=(data_t scalar) override;


         /// assign a scalar to all elements of the data vector
         DataHandler<data_t>& operator=(data_t scalar) override;



     protected:
         /// the vector storing the data
         DataVector_t _data;

         /// implement the polymorphic clone operation
         DataHandlerCPU<data_t>* cloneImpl() const override;

         /// implement the polymorphic comparison operation
         bool isEqual(const DataHandler<data_t>& other) const override;
     };

} // namespace elsa
