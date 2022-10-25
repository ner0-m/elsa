#pragma once

#include "LinearOperator.h"
#include "BlockDescriptor.h"

#include <vector>

namespace elsa
{
    /**
     * @brief Class representing a block operator matrix
     *
     * @author Matthias Wieczorek - initial code
     * @author David Frank - rewrite
     * @author Nikola Dinev - automatic descriptor generation, rewrite
     *
     * @tparam data_t  data type for the domain and range of the operator, defaulting to real_t
     *
     * A block linear operator represents a block operator matrix \f$ B \f$ consisting of multiple
     * matrices \f$ A_i, i=1,\ldots,n \f$ stacked:
     * - rowwise \f[ B = \begin{bmatrix}
     *                      A_{1}\\
     *                      A_{2}\\
     *                      \vdots\\
     *                      A_{n}
     *                   \end{bmatrix} \f]
     *
     * - columnwise  \f[ B = \begin{bmatrix} A_{1} & A_{2} & \hdots & A_{n} \end{bmatrix} \f]
     */
    template <typename data_t = real_t>
    class BlockLinearOperator : public LinearOperator<data_t>
    {
    public:
        /// possible arrangements of the blocks
        enum BlockType {
            ROW,
            COL,
        };

        /// convenience typedef for a vector of pointers to LinearOperator
        using OperatorList = typename std::vector<std::unique_ptr<LinearOperator<data_t>>>;

        /**
         * @brief Construct a BlockLinearOperator of the given BlockType from the list of operators
         *
         * @param[in] ops the list of operators
         * @param[in] blockType the fashion in which the operators are to be stacked
         *
         * @throw InvalidArgumentError if ops is empty
         *
         * The domain and range descriptors of the BlockLinearOperator are generated automatically
         * based on the descriptors of the operators in the list. For the block descriptor, a
         * PartitionDescriptor is preferentially generated, if not possible a RandomBlocksDescriptor
         * is generated instead. For the non-block descriptor the best common descriptor is chosen
         * (see DataDescriptor::bestCommon()).
         */
        BlockLinearOperator(const OperatorList& ops, BlockType blockType);

        /**
         * @brief Construct a BlockLinearOperator of the given BlockType from the list of operators,
         * and additionally manually set the domain and range descriptors of the operator
         *
         * @param[in] domainDescriptor descriptor of the domain of the operator
         * @param[in] rangeDescriptor descriptor of the range of the operator
         * @param[in] ops the list of operators
         * @param[in] blockType the fashion in which the operators are to be stacked
         *
         * @throw InvalidArgumentError if the passed in descriptors are not suitable for the
         * BlockLinearOperator
         */
        BlockLinearOperator(const DataDescriptor& domainDescriptor,
                            const DataDescriptor& rangeDescriptor, const OperatorList& ops,
                            BlockType blockType);

        /// default destructor
        ~BlockLinearOperator() override = default;

        BlockLinearOperator& operator=(BlockLinearOperator&) = delete;

        /// return the operator corresponding to the i-th block of the matrix
        const LinearOperator<data_t>& getIthOperator(index_t i) const;

        /// return the total number of blocks
        index_t numberOfOps() const;

        /// apply the block linear operator
        void apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the block linear operator
        void applyAdjoint(const DataContainer<data_t>& y,
                          DataContainer<data_t>& Aty) const override;

        // Pull in apply and applyAdjoint with single argument from base class
        using LinearOperator<data_t>::apply;
        using LinearOperator<data_t>::applyAdjoint;

    protected:
        /// protected copy constructor; used for cloning
        BlockLinearOperator(const BlockLinearOperator& other);

        /// implement the polymorphic clone operation
        BlockLinearOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// list specifying the individual operators corresponding to each block
        OperatorList _operatorList;

        /// determines in which fashion the operators are concatenated - rowwise or columnwise
        BlockType _blockType;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;

        /// returns the best fitting domain descriptor based on the operator list and block type
        static std::unique_ptr<DataDescriptor>
            determineDomainDescriptor(const OperatorList& operatorList, BlockType blockType);

        /// returns the best fitting range descriptor based on the operator list and block type
        static std::unique_ptr<DataDescriptor>
            determineRangeDescriptor(const OperatorList& operatorList, BlockType blockType);

        /// finds the best fitting block descriptor, such that each block is described by the
        /// corresponding descriptor in the list
        static std::unique_ptr<BlockDescriptor>
            bestBlockDescriptor(const std::vector<const DataDescriptor*>&);
    };
} // namespace elsa
