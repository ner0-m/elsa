#pragma once

#include <memory>
#include <iostream>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "DnnlLayer.h"

#include "dnnl.hpp"

namespace elsa
{
    enum class Loss { MeanSquareError, CrossEntropy };

    template <typename data_t>
    class DnnlMeanSquareError;

    template <typename data_t>
    class DnnlCrossEntropy;

    template <typename data_t>
    class DnnlLoss
    {
    public:
        static std::unique_ptr<DnnlLoss<data_t>>
            createDnnlLoss(const DataDescriptor& inputDescriptor, Loss loss);

        DnnlLoss() = default;

        virtual ~DnnlLoss() = default;

        explicit DnnlLoss(const DataDescriptor& inputDescriptor);

        /// Evaluate loss and its gradient between prediction and label,
        virtual void evaluate(const DataContainer<data_t>& prediction,
                              const DataContainer<data_t>& label);

        void setEngine(std::shared_ptr<dnnl::engine> engine);

        /// Get loss
        data_t getLoss() const;

        /// Get loss gradient
        std::shared_ptr<dnnl::memory> getLossGradientMemory() const;

    protected:
        std::unique_ptr<DataDescriptor> _inputDescriptor;

        data_t _loss;
        std::shared_ptr<dnnl::memory> _lossGradient = nullptr;

        std::shared_ptr<dnnl::engine> _engine = nullptr;
    };

    template <typename data_t>
    inline std::unique_ptr<DnnlLoss<data_t>>
        DnnlLoss<data_t>::createDnnlLoss(const DataDescriptor& inputDescriptor, Loss loss)
    {
        switch (loss) {
            case Loss::MeanSquareError:
                return std::make_unique<DnnlMeanSquareError<data_t>>(inputDescriptor);
            case Loss::CrossEntropy:
                return std::make_unique<DnnlCrossEntropy<data_t>>(inputDescriptor);
            default:
                throw std::invalid_argument("Unkown loss function");
        }
    }

    template <typename data_t>
    DnnlLoss<data_t>::DnnlLoss(const DataDescriptor& inputDescriptor)
        : _inputDescriptor(inputDescriptor.clone())
    {
    }

    template <typename data_t>
    data_t DnnlLoss<data_t>::getLoss() const
    {
        return _loss;
    }

    template <typename data_t>
    std::shared_ptr<dnnl::memory> DnnlLoss<data_t>::getLossGradientMemory() const
    {
        return _lossGradient;
    }

    template <typename data_t>
    void DnnlLoss<data_t>::setEngine(std::shared_ptr<dnnl::engine> engine)
    {
        _engine = engine;
    }

    template <typename data_t>
    void DnnlLoss<data_t>::evaluate(const DataContainer<data_t>& prediction,
                                    const DataContainer<data_t>& label)
    {
        if (*_inputDescriptor != label.getDataDescriptor()
            || *_inputDescriptor != prediction.getDataDescriptor())
            throw std::invalid_argument("Input descriptor and descriptor of label must match");

        if (!_engine)
            throw std::logic_error("Cannot evaluate loss: Dnnl engine is null.");

        if (!_lossGradient) {
            dnnl::memory::dims dims;
            for (const auto& dim :
                 prediction.getDataDescriptor().getNumberOfCoefficientsPerDimension())
                dims.push_back(dim);

            auto chooseFormatTag = [](index_t dim) {
                using ft = dnnl::memory::format_tag;
                switch (dim) {
                    case 2:
                        return ft::nc;
                    case 3:
                        return ft::ncw;
                    case 4:
                        return ft::nchw;
                    case 5:
                        return ft::ncdhw;
                    default:
                        return ft::undef;
                }
            };

            auto desc =
                dnnl::memory::desc({dims}, detail::TypeToDnnlTypeTag<data_t>::tag,
                                   chooseFormatTag(_inputDescriptor->getNumberOfDimensions()));
            if (!_lossGradient)
                _lossGradient = std::make_shared<dnnl::memory>(desc, *_engine);
        }
    }

    template <typename data_t>
    class DnnlMeanSquareError : public DnnlLoss<data_t>
    {
    public:
        using BaseType = DnnlLoss<data_t>;

        explicit DnnlMeanSquareError(const DataDescriptor& inputDescriptor);

        void evaluate(const DataContainer<data_t>& prediction,
                      const DataContainer<data_t>& label) override;

    private:
        using BaseType::_inputDescriptor;
        using BaseType::_loss;
        using BaseType::_lossGradient;
        using BaseType::_engine;
    };

    template <typename data_t>
    class DnnlCrossEntropy : public DnnlLoss<data_t>
    {
    public:
        using BaseType = DnnlLoss<data_t>;

        explicit DnnlCrossEntropy(const DataDescriptor& inputDescriptor);

        void evaluate(const DataContainer<data_t>& prediction,
                      const DataContainer<data_t>& label) override;

    private:
        using BaseType::_inputDescriptor;
        using BaseType::_loss;
        using BaseType::_lossGradient;
        using BaseType::_engine;
    };

} // namespace elsa
