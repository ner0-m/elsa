#pragma once

#include "LinearOperator.h"

#include <vector>

namespace elsa
{
    /**
     * \brief Operator to compute finite differences.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - rewrite and performance optimization
     * \author Tobias Lasser - modernization
     *
     * \tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * This class represents a linear operator D that computes finite differences,
     * using the central, forward, or backward differences.
     */
    template <typename data_t = real_t>
    class FiniteDifferences : public LinearOperator<data_t>
    {
    public:
        /// supported types of finite differences
        enum class DiffType { FORWARD, BACKWARD, CENTRAL };

        /**
         * \brief Constructor for FiniteDifferences over all dimensions.
         *
         * \param[in] domainDescriptor Descriptor for domain
         * \param[in] type denoting the type of finite differences
         *
         * This implementation uses zero padding such that it's equal to the
         * following matrix formulation (in 1D)
         * - Dforward  = full(spdiags([-e e], 0:1, n, n));
         * - Dbackward = full(spdiags([-e e], -1:0, n, n));
         * - Dcentral  = spdiags(0.5*[-e e], [-1,1], n, n);
         *
         * Note: the descriptor for the range is automatically generated from the domain.
         */
        explicit FiniteDifferences(const DataDescriptor& domainDescriptor,
                                   DiffType type = DiffType::FORWARD);

        /**
         * \brief Constructor for FiniteDifferences over selected dimensions.
         *
         * \param[in] domainDescriptor Descriptor for domain
         * \param[in] activeDims Boolean vector defining which dimensions are active or not
         * \param[in] type denoting the type of finite differences
         *
         * This implementation uses zero padding such that it's equal to the
         * following matrix formulation (in 1D)
         * - Dforward  = full(spdiags([-e e], 0:1, n, n));
         * - Dbackward = full(spdiags([-e e], -1:0, n, n));
         * - Dcentral  = spdiags(0.5*[-e e], [-1,1], n, n);
         *
         * Note: the descriptor for the range is automatically generated from the domain.
         */
        FiniteDifferences(const DataDescriptor& domainDescriptor, const BooleanVector_t& activeDims,
                          DiffType type = DiffType::FORWARD);

        /// default destructor
        ~FiniteDifferences() override = default;

    protected:
        /// apply the finite differences operator
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the finite differences operator
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        FiniteDifferences<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// type of the finite differences to be computed
        DiffType _type;

        /// boolean vector for active dimensions when computing finite differences
        BooleanVector_t _activeDims;

        /// precompute some helper variables to optimize speed
        void precomputeHelpers();

        std::vector<index_t> _coordDiff{};  /// precomputed helper for coordinate diffs
        std::vector<index_t> _coordDelta{}; /// precomputed helper for coordinate deltas
        std::vector<index_t> _dimCounter{}; /// precomputed helper for active dim counter

        /// the actual finite differences computations (with mode as template parameter for
        /// performance)
        template <typename FDtype>
        void applyHelper(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                         FDtype type) const;

        /// the actual finite differences computations (with mode as template parameter for
        /// performance)
        template <typename FDtype>
        void applyAdjointHelper(const DataContainer<data_t>& y, DataContainer<data_t>& Aty,
                                FDtype type) const;
    };
} // namespace elsa
