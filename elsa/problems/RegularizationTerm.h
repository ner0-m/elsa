#pragma once

#include "Functional.h"

#include <memory>

namespace elsa
{
    /**
     * \brief Class representing a regularization term (a scalar parameter and a functional).
     *
     * \author Maximilian Hornung - initial code
     * \author Tobias Lasser - modernization
     *
     * \tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a regularization term, consisting of a regularization parameter (a
     * scalar) and a term (represented as a Functional). It is just a light-weight wrapper with no
     * added functionality.
     */
    template <typename data_t = real_t>
    class RegularizationTerm
    {
    public:
        /**
         * \brief Constructor for the regularization term, accepting a weight and a functional.
         *
         * \param weight the regularization parameter
         * \param functional the actual term
         */
        RegularizationTerm(data_t weight, const Functional<data_t>& functional);

        /// copy constructor
        RegularizationTerm(const RegularizationTerm<data_t>& other);

        /// copy assignment
        RegularizationTerm<data_t>& operator=(const RegularizationTerm<data_t>& other);

        /// move constructor
        RegularizationTerm(RegularizationTerm<data_t>&& other) noexcept;

        /// move assignment
        RegularizationTerm<data_t>& operator=(RegularizationTerm<data_t>&& other) noexcept;

        /// the default destructor
        ~RegularizationTerm() = default;

        /// return the weight of the regularization term (the regularization parameter)
        data_t getWeight() const;

        /// return the functional of the regularization term
        Functional<data_t>& getFunctional() const;

        /// comparison operator
        bool operator==(const RegularizationTerm<data_t>& other) const;

        /// negative comparison operator
        bool operator!=(const RegularizationTerm<data_t>& other) const;

    private:
        /// the weight / regularization parameter
        data_t _weight;

        /// the functional of the regularization term
        std::unique_ptr<Functional<data_t>> _functional;
    };
} // namespace elsa
