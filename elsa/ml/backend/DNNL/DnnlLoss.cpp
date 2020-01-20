#include "DnnlLoss.h"

#include <limits>

namespace elsa
{
    template <typename data_t>
    DnnlMeanSquareError<data_t>::DnnlMeanSquareError(const DataDescriptor& inputDescriptor)
        : DnnlLoss<data_t>(inputDescriptor)
    {
    }

    template <typename data_t>
    void DnnlMeanSquareError<data_t>::evaluate(const DataContainer<data_t>& prediction,
                                               const DataContainer<data_t>& label)
    {
        BaseType::evaluate(prediction, label);

        index_t size = _inputDescriptor->getNumberOfCoefficients();

        Eigen::VectorX<data_t> predictionVec(size);
        Eigen::VectorX<data_t> labelVec(size);

        for (index_t i = 0; i < size; ++i) {
            labelVec.coeffRef(i) = label[i];
            predictionVec.coeffRef(i) = prediction[i];
        }

        DataContainer<data_t> tmp = prediction - label;

        _loss = (static_cast<data_t>(1) / static_cast<data_t>(size)) * tmp.squaredL2Norm();

        Eigen::Map<Eigen::VectorX<data_t>> lossGradientMap(
            static_cast<data_t*>(_lossGradient->get_data_handle()), size);

        lossGradientMap = -(static_cast<data_t>(1) / static_cast<data_t>(size)) * 2
                          * (labelVec.array() - predictionVec.array()).matrix();
    }

    template <typename data_t>
    DnnlCrossEntropy<data_t>::DnnlCrossEntropy(const DataDescriptor& inputDescriptor)
        : DnnlLoss<data_t>(inputDescriptor)
    {
    }

    template <typename data_t>
    void DnnlCrossEntropy<data_t>::evaluate(const DataContainer<data_t>& prediction,
                                            const DataContainer<data_t>& label)
    {
        BaseType::evaluate(prediction, label);

        index_t size = _inputDescriptor->getNumberOfCoefficients();

        Eigen::VectorX<data_t> predictionVec(size);
        Eigen::VectorX<data_t> labelVec(size);

        for (index_t i = 0; i < size; ++i) {
            labelVec.coeffRef(i) = label[i];
            predictionVec.coeffRef(i) = prediction[i];
        }

        DataContainer<data_t> tmp = prediction - label;

        _loss = -labelVec.dot(predictionVec.cwiseMax(1e-15).array().log().matrix());

        Eigen::Map<Eigen::VectorX<data_t>> lossGradientMap(
            static_cast<data_t*>(_lossGradient->get_data_handle()), size);

        lossGradientMap = -(labelVec.array() * (1 / predictionVec.array())).matrix();
    }

    template class DnnlMeanSquareError<float>;
    template class DnnlCrossEntropy<float>;
} // namespace elsa