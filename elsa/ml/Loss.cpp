#include "Loss.h"

namespace elsa::ml
{
    // We assume the batch-size (along which we will block) is the last dimension
    static IdenticalBlocksDescriptor getBlockedBatchDescriptor(const DataDescriptor& desc)
    {
        index_t batchSize = desc.getNumberOfCoefficientsPerDimension().tail(1)(0);
        IndexVector_t blockDims = desc.getNumberOfCoefficientsPerDimension().head(
            desc.getNumberOfCoefficientsPerDimension().size() - 1);
        return IdenticalBlocksDescriptor(batchSize, VolumeDescriptor(blockDims));
    }

    template <typename data_t>
    static std::pair<index_t, index_t> getSizeParameters(const DataContainer<data_t>& x)
    {
        // As always we assume the batch-size to be the last dimension
        index_t batchSize = x.getDataDescriptor().getNumberOfCoefficientsPerDimension().tail(1)(0);

        index_t size = x.getDataDescriptor().getNumberOfCoefficients();

        return std::make_pair<index_t, index_t>(std::move(size), std::move(batchSize));
    }

    template <typename data_t>
    static data_t reduceLoss(LossReduction reduction, const std::vector<data_t>& batchLoss)
    {
        switch (reduction) {
            case LossReduction::SumOverBatchSize:
                return std::accumulate(batchLoss.begin(), batchLoss.end(), data_t(0))
                       / static_cast<data_t>(batchLoss.size());
            case LossReduction::Sum:
                return std::accumulate(batchLoss.begin(), batchLoss.end(), data_t(0));
            default:
                throw std::invalid_argument("Unknown loss-reduction");
        }
    }

    template <typename data_t>
    static DataContainer<data_t> unreduceGradient(LossReduction reduction,
                                                  const DataContainer<data_t>& gradient)
    {
        if (reduction == LossReduction::SumOverBatchSize) {
            return gradient / gradient.getDataDescriptor().getNumberOfCoefficientsPerDimension()(1);
        }
        return gradient;
    }

    template <typename data_t>
    Loss<data_t>::Loss(LossReduction reduction, const std::string& name)
        : reduction_(reduction), name_(name)
    {
    }
    template <typename data_t>
    data_t Loss<data_t>::getLoss(const DataContainer<data_t>& x,
                                 const DataContainer<data_t>& y) const
    {
        return lossFunction_(reduction_, x, y);
    }

    template <typename data_t>
    DataContainer<data_t> Loss<data_t>::getLossGradient(const DataContainer<data_t>& x,
                                                        const DataContainer<data_t>& y) const
    {
        return lossGradientFunction_(reduction_, x, y);
    }

    template <typename data_t>
    data_t Loss<data_t>::operator()(const DataContainer<data_t>& x, const DataContainer<data_t>& y)
    {
        return getLoss(x, y);
    }

    template <typename data_t>
    std::string Loss<data_t>::getName() const
    {
        return name_;
    }

    template <typename data_t>
    BinaryCrossentropy<data_t>::BinaryCrossentropy(LossReduction reduction)
        : Loss<data_t>(reduction, "BinaryCrossentropy")
    {
        this->lossFunction_ = &BinaryCrossentropy<data_t>::lossImpl;
        this->lossGradientFunction_ = &BinaryCrossentropy<data_t>::lossGradientImpl;
    }

    template <typename data_t>
    data_t BinaryCrossentropy<data_t>::lossImpl(LossReduction reduction,
                                                const DataContainer<data_t>& x,
                                                const DataContainer<data_t>& y)
    {
        // Get blocked descriptor where each block represents a single batch
        auto batchDesc = getBlockedBatchDescriptor(x.getDataDescriptor());

        std::vector<data_t> batchLoss(asIndex(batchDesc.getNumberOfBlocks()), data_t(0));

        // Calulate binary-crossentropy for each batch
        for (index_t b = 0; b < batchDesc.getNumberOfBlocks(); ++b) {
#ifndef ELSA_CUDA_VECTOR
            auto x_expr = (data_t(1) * x.viewAs(batchDesc).getBlock(b)).eval().array();
            auto y_expr = (data_t(1) * y.viewAs(batchDesc).getBlock(b)).eval().array();
            batchLoss[asIndex(b)] =
                (y_expr * x_expr.max(std::numeric_limits<data_t>::epsilon()).log()
                 + (1 - y_expr) * (1 - x_expr).max(std::numeric_limits<data_t>::epsilon()).log())
                    .mean();
#else
            DataContainer<data_t> x_expr = x.viewAs(batchDesc).getBlock(b);
            DataContainer<data_t> x2_expr = 1 - x_expr;
            DataContainer<data_t> y_expr = y.viewAs(batchDesc).getBlock(b);

            for (index_t i = 0; i < x_expr.getSize(); ++i) {
                x_expr[i] = std::max(x_expr[i], std::numeric_limits<data_t>::epsilon());
                x2_expr[i] = std::max(x2_expr[i], std::numeric_limits<data_t>::epsilon());
            }
            DataContainer<data_t> l = y_expr * log(x_expr) + (1 - y_expr) * log(x2_expr);
            batchLoss[asIndex(b)] = l.sum() / x_expr.getSize();
#endif
        }

        // reduce loss
        data_t loss = reduceLoss(reduction, batchLoss);

        loss *= data_t(-1);
        return loss;
    }

    template <typename data_t>
    DataContainer<data_t> BinaryCrossentropy<data_t>::lossGradientImpl(
        LossReduction reduction, const DataContainer<data_t>& x, const DataContainer<data_t>& y)
    {
#ifndef ELSA_CUDA_VECTOR
        auto x_expr = (data_t(1) * x).eval().array().max(std::numeric_limits<data_t>::epsilon());
        auto y_expr = (data_t(1) * y).eval().array();
        Eigen::VectorXf data =
            data_t(-1) / data_t(2)
            * (y_expr * data_t(1) / x_expr
               + (data_t(1) - y_expr) * data_t(1)
                     / (data_t(1) - x_expr).max(std::numeric_limits<data_t>::epsilon()));
        return unreduceGradient(reduction, DataContainer<data_t>(x.getDataDescriptor(), data));
#else
        DataContainer<data_t> x_expr = x;
        DataContainer<data_t> x2_expr = 1 - x;
        for (index_t i = 0; i < x_expr.getSize(); ++i) {
            x_expr[i] = std::max(x_expr[i], std::numeric_limits<data_t>::epsilon());
            x2_expr[i] = std::max(x2_expr[i], std::numeric_limits<data_t>::epsilon());
        }
        DataContainer<data_t> data = y / x_expr + (1 - y) / x2_expr;
        data *= data_t(-1) / data_t(2);
        return unreduceGradient(reduction, data.viewAs(x.getDataDescriptor()));
#endif
    }

    template <typename data_t>
    CategoricalCrossentropy<data_t>::CategoricalCrossentropy(LossReduction reduction)
        : Loss<data_t>(reduction, "CategoricalCrossentropy")
    {
        this->lossFunction_ = &CategoricalCrossentropy<data_t>::lossImpl;
        this->lossGradientFunction_ = &CategoricalCrossentropy<data_t>::lossGradientImpl;
    }

    template <typename data_t>
    data_t CategoricalCrossentropy<data_t>::lossImpl(LossReduction reduction,
                                                     const DataContainer<data_t>& x,
                                                     const DataContainer<data_t>& y)
    {
        // Get blocked descriptor where each block represents a single batch
        auto batchDesc = getBlockedBatchDescriptor(x.getDataDescriptor());

        // Calculate loss for each batch
        std::vector<data_t> batchLoss(asIndex(batchDesc.getNumberOfBlocks()), data_t(0));
        for (int b = 0; b < batchDesc.getNumberOfBlocks(); ++b) {
#ifndef ELSA_CUDA_VECTOR
            auto x_expr = (data_t(1) * x.viewAs(batchDesc).getBlock(b))
                              .eval()
                              .array()
                              .max(std::numeric_limits<data_t>::epsilon());
            auto y_expr = (data_t(1) * y.viewAs(batchDesc).getBlock(b)).eval();
            batchLoss[asIndex(b)] = y_expr.dot(x_expr.log().matrix());
#else
            DataContainer<data_t> x_expr = x.viewAs(batchDesc).getBlock(b);
            for (index_t i = 0; i < x_expr.getSize(); ++i) {
                x_expr[i] = std::max(x_expr[i], std::numeric_limits<data_t>::epsilon());
            }
            DataContainer<data_t> y_expr = y.viewAs(batchDesc).getBlock(b);
            batchLoss[asIndex(b)] = y_expr.dot(log(x_expr));
#endif
        }
        data_t loss = reduceLoss(reduction, batchLoss);
        loss *= data_t(-1);
        return loss;
    }

    template <typename data_t>
    DataContainer<data_t> CategoricalCrossentropy<data_t>::lossGradientImpl(
        LossReduction reduction, const DataContainer<data_t>& x, const DataContainer<data_t>& y)
    {
#ifndef ELSA_CUDA_VECTOR
        auto x_expr = (data_t(1) * x).eval().array().max(std::numeric_limits<data_t>::epsilon());
        auto y_expr = (data_t(1) * y).eval().array();
        Eigen::VectorXf data = -data_t(1) * (y_expr * data_t(1) / x_expr);
        return unreduceGradient(reduction, DataContainer<data_t>(y.getDataDescriptor(), data));
#else
        DataContainer<data_t> x_expr = x;
        for (index_t i = 0; i < x.getSize(); ++i) {
            x_expr[i] = std::max(x_expr[i], std::numeric_limits<data_t>::epsilon());
        }
        DataContainer<data_t> data = -data_t(1) * y / x_expr;
        return unreduceGradient(reduction, data.viewAs(y.getDataDescriptor()));
#endif
    }

    template <typename data_t>
    SparseCategoricalCrossentropy<data_t>::SparseCategoricalCrossentropy(LossReduction reduction)
        : Loss<data_t>(reduction, "SparseCategoricalCrossentropy")
    {
        this->lossFunction_ = &SparseCategoricalCrossentropy<data_t>::lossImpl;
        this->lossGradientFunction_ = &SparseCategoricalCrossentropy<data_t>::lossGradientImpl;
    }

    template <typename data_t>
    data_t SparseCategoricalCrossentropy<data_t>::lossImpl(LossReduction reduction,
                                                           const DataContainer<data_t>& x,
                                                           const DataContainer<data_t>& y)
    {
        // This loss is the same as CategoricalCrossentropy but doesn't require
        // one-hot encoded labels. We therefore translate all labels to one-hot
        // and call CategoricalCrossentropy.

        // x has shape (num_classes, batch_size)
        index_t numClasses = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()(0);
        index_t batchSize = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()(1);

        return CategoricalCrossentropy<data_t>(reduction)(
            x, Utils::Encoding::toOneHot(y, numClasses, batchSize));
    }

    template <typename data_t>
    DataContainer<data_t> SparseCategoricalCrossentropy<data_t>::lossGradientImpl(
        LossReduction reduction, const DataContainer<data_t>& x, const DataContainer<data_t>& y)
    {
        // x has shape (num_classes, batch_size)
        index_t numClasses = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()(0);
        index_t batchSize = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()(1);

        DataContainer<data_t> oneHot = Utils::Encoding::toOneHot(y, numClasses, batchSize);
#ifndef ELSA_CUDA_VECTOR
        auto x_expr = (data_t(1) * x).eval().array().max(std::numeric_limits<data_t>::epsilon());
        auto y_expr = (data_t(1) * oneHot).eval().array();
        Eigen::VectorXf data = -data_t(1) * (y_expr * data_t(1) / x_expr);
        return unreduceGradient(reduction, DataContainer<data_t>(oneHot.getDataDescriptor(), data));
#else
        DataContainer<data_t> x_expr = x;
        for (index_t i = 0; i < x_expr.getSize(); ++i) {
            x_expr[i] = std::max(x_expr[i], std::numeric_limits<data_t>::epsilon());
        }
        DataContainer<data_t> data = -data_t(1) * (oneHot / x_expr);
        return unreduceGradient(reduction, data.viewAs(oneHot.getDataDescriptor()));
#endif
    }

    template <typename data_t>
    MeanSquaredError<data_t>::MeanSquaredError(LossReduction reduction)
        : Loss<data_t>(reduction, "MeanSquaredError")
    {
        this->lossFunction_ = &MeanSquaredError<data_t>::lossImpl;
        this->lossGradientFunction_ = &MeanSquaredError<data_t>::lossGradientImpl;
    }

    template <typename data_t>
    data_t MeanSquaredError<data_t>::lossImpl(LossReduction reduction,
                                              const DataContainer<data_t>& x,
                                              const DataContainer<data_t>& y)
    {

        // Get blocked descriptor where each block represents a single batch
        auto batchDesc = getBlockedBatchDescriptor(x.getDataDescriptor());

        // Calculate loss for each batch
        std::vector<data_t> batchLoss(asIndex(batchDesc.getNumberOfBlocks()), data_t(0));
        for (index_t b = 0; b < batchDesc.getNumberOfBlocks(); ++b) {
#ifndef ELSA_CUDA_VECTOR
            auto x_expr = (data_t(1) * x.viewAs(batchDesc).getBlock(b)).eval().array();
            auto y_expr = (data_t(1) * y.viewAs(batchDesc).getBlock(b)).eval().array();
            batchLoss[asIndex(b)] = ((y_expr - x_expr) * (y_expr - x_expr)).mean();
#else
            DataContainer<data_t> x_expr = x.viewAs(batchDesc).getBlock(b);
            DataContainer<data_t> y_expr = y.viewAs(batchDesc).getBlock(b);
            DataContainer<data_t> l = ((y_expr - x_expr) * (y_expr - x_expr));
            batchLoss[asIndex(b)] = l.sum() / x_expr.getSize();
#endif
        }
        data_t loss = reduceLoss(reduction, batchLoss);
        return loss;
    }

    template <typename data_t>
    DataContainer<data_t> MeanSquaredError<data_t>::lossGradientImpl(LossReduction reduction,
                                                                     const DataContainer<data_t>& x,
                                                                     const DataContainer<data_t>& y)
    {
        DataContainer<data_t> gradient =
            data_t(2)
            / static_cast<data_t>(x.getDataDescriptor().getNumberOfCoefficientsPerDimension()(0))
            * (y - x);
        return unreduceGradient(reduction, gradient);
    }

    template class Loss<float>;
    template class BinaryCrossentropy<float>;
    template class CategoricalCrossentropy<float>;
    template class SparseCategoricalCrossentropy<float>;
    template class MeanSquaredError<float>;

} // namespace elsa::ml
