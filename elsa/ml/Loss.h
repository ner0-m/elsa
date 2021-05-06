#pragma once

#include <numeric>
#include <vector>
#include <functional>
#include <algorithm>
#include <string>

#include "DataContainer.h"
#include "IdenticalBlocksDescriptor.h"
#include "Common.h"
#include "Utils.h"

namespace elsa::ml
{
    /// Reduction types for loss functions
    enum class LossReduction {
        /// reduce loss by summing up across batches
        Sum,
        /// reduce loss by summing up over batches
        SumOverBatchSize
    };

    /// Base class for all loss functions
    ///
    /// @author David Tellenbach
    template <typename data_t = real_t>
    class Loss
    {
    public:
        /// default constructor
        Loss() = default;

        /// virtual destructor
        virtual ~Loss() = default;

        /// @returns the loss between predictions x and labels y
        data_t getLoss(const DataContainer<data_t>& x, const DataContainer<data_t>& y) const;

        /// @returns the loss between predictions x and labels y
        data_t operator()(const DataContainer<data_t>& x, const DataContainer<data_t>& y);

        /// @returns the loss-gradient between predictions x and labels y
        DataContainer<data_t> getLossGradient(const DataContainer<data_t>& x,
                                              const DataContainer<data_t>& y) const;

        /// @return the name of this loss function
        std::string getName() const;

    protected:
        Loss(LossReduction reduction, const std::string& name);

        using LossFunctionType = std::function<data_t(LossReduction, DataContainer<data_t> const&,
                                                      DataContainer<data_t> const&)>;

        using LossGradientFunctionType = std::function<DataContainer<data_t>(
            LossReduction, DataContainer<data_t> const&, DataContainer<data_t> const&)>;

        LossFunctionType lossFunction_;
        LossGradientFunctionType lossGradientFunction_;
        LossReduction reduction_;
        std::string name_;
    };

    /// @brief Computes the cross-entropy loss between true labels and predicted
    /// labels.
    ///
    /// @author David Tellenbach
    ///
    /// Use this cross-entropy loss when there are only two label classes
    /// (assumed to be 0 and 1). For each example, there should be a single
    /// floating-point value per prediction.
    template <typename data_t = real_t>
    class BinaryCrossentropy : public Loss<data_t>
    {
    public:
        /// Construct a BinaryCrossEntropy loss by optionally specifying the
        /// reduction type.
        explicit BinaryCrossentropy(LossReduction reduction = LossReduction::SumOverBatchSize);

    private:
        static data_t lossImpl(LossReduction, const DataContainer<data_t>&,
                               const DataContainer<data_t>&);

        static DataContainer<data_t> lossGradientImpl(LossReduction, const DataContainer<data_t>&,
                                                      const DataContainer<data_t>&);
    };

    /// @brief Computes the crossentropy loss between the labels and predictions.
    ///
    /// @author David Tellenbach
    ///
    /// Use this crossentropy loss function when there are two or more label
    /// classes. We expect labels to be provided in a one-hot representation.
    /// If you don't want to use one-hot encoded labels, please use
    /// SparseCategoricalCrossentropy loss. There should be # classes floating
    /// point values per feature.
    template <typename data_t = real_t>
    class CategoricalCrossentropy : public Loss<data_t>
    {
    public:
        /// Construct a CategoricalCrossentropy loss by optionally specifying the
        /// reduction type.
        explicit CategoricalCrossentropy(LossReduction reduction = LossReduction::SumOverBatchSize);

    private:
        static data_t lossImpl(LossReduction reduction, const DataContainer<data_t>& x,
                               const DataContainer<data_t>& y);

        static DataContainer<data_t> lossGradientImpl(LossReduction reduction,
                                                      const DataContainer<data_t>& x,
                                                      const DataContainer<data_t>& y);
    };

    /// @brief Computes the crossentropy loss between the labels and predictions.
    ///
    /// @author David Tellenbach
    ///
    /// Use this crossentropy loss function when there are two or more label
    /// classes. If you want to provide labels using one-hot representation,
    /// please use CategoricalCrossentropy loss. There should be # classes
    /// floating point values per feature for x and a single floating point
    /// value per feature for y.
    template <typename data_t = real_t>
    class SparseCategoricalCrossentropy : public Loss<data_t>
    {
    public:
        /// Construct a SparseCategoricalCrossentropy loss by optionally specifying the
        /// reduction type.
        explicit SparseCategoricalCrossentropy(
            LossReduction reduction = LossReduction::SumOverBatchSize);

    private:
        static data_t lossImpl(LossReduction, const DataContainer<data_t>&,
                               const DataContainer<data_t>&);

        static DataContainer<data_t> lossGradientImpl(LossReduction reduction,
                                                      const DataContainer<data_t>& x,
                                                      const DataContainer<data_t>& y);
    };

    /// @brief Computes the mean squared error between labels y and predictions x:
    ///
    /// @author David Tellenbach
    ///
    /// \f[ \text{loss} = (y - x)^2 \f]
    template <typename data_t = real_t>
    class MeanSquaredError : public Loss<data_t>
    {
    public:
        /// Construct a MeanSquaredError loss by optionally specifying the
        /// reduction type.
        explicit MeanSquaredError(LossReduction reduction = LossReduction::SumOverBatchSize);

    private:
        static data_t lossImpl(LossReduction, const DataContainer<data_t>&,
                               const DataContainer<data_t>&);

        static DataContainer<data_t> lossGradientImpl(LossReduction, const DataContainer<data_t>&,
                                                      const DataContainer<data_t>&);
    };

} // namespace elsa::ml
