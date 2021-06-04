#pragma once

#include "Model.h"
#include "Common.h"
#include "Layer.h"

namespace elsa::ml
{
    /**
     * @brief Class representing the PhantomNet model
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t The type of all coefficients used in this model. This parameter is optional
     * and defaults to real_t.
     * @tparam Backend The MlBackend that will be used for inference and training. This parameter is
     * optional and defaults to MlBackend::Auto. // TODO Auto or Dnnl?
     *
     * References:
     * https://arxiv.org/pdf/1811.04602.pdf
     */
    template <typename data_t = real_t, MlBackend Backend = MlBackend::Dnnl>
    class PhantomNet : public Model<data_t, Backend>
    {
        // TODO check all these
    public:
        PhantomNet() = default;

        template <typename FirstLayerType, typename... LayerTypes>
        PhantomNet(FirstLayerType&& input, LayerTypes&&... layers)
        {
            addLayerHelper(std::forward<FirstLayerType>(input),
                           std::forward<LayerTypes>(layers)...);
            assert(!layers_.empty());
            this->outputs_.push_back(layers_.back());
            this->batchSize_ = this->inputs_.front()->getBatchSize();
            this->setInputDescriptors();
        }

        template <typename LayerType>
        void add(LayerType&& layer)
        {
            if constexpr (std::is_same_v<std::decay_t<LayerType>, Input<data_t>>) {
                if (layers_.size() != 0) {
                    throw std::invalid_argument(
                        "Only first layer of a sequential model can be Input");
                }
                Input<data_t>* input = new LayerType(std::move(layer));
                this->inputs_.push_back(input);
                layers_.push_back(input);
            } else {
                if (!layers_.size()) {
                    throw std::invalid_argument(
                        "First layer of a sequential model must be an input layer");
                }
                Layer<data_t>* nextLayer = new LayerType(std::move(layer));
                nextLayer->setInput(layers_.back());
                layers_.push_back(nextLayer);
            }
        }

        // TODO what about these two?

        /// make copy constructor deletion explicit
        PhantomNet(const PhantomNet<data_t>&) = delete;

        /// default destructor
        //        ~PhantomNet() = default;
        ~PhantomNet()
        {
            for (auto& layer : layers_)
                delete layer;
        }

        // TODO considered having these 3 layers in the ml package alongside other Layers e.g.
        //  Softmax, UpSampling and such, but given that these 3 are not quite conventional,
        //  following this approach, that package would be overpopulated with layers that would
        //  essentially be used once, even more so with more incoming models

        /**
         * @brief Class representing the Trimmed-DenseBlock layer in the PhantomNet model
         *
         * @author Andi Braimllari - initial code
         *
         * References:
         * https://arxiv.org/pdf/1811.04602.pdf
         */
        // TODO or should it inherit from Trainable, or even ConvBase(have a look at Dense as well)
        //  probably just Trainable
        class TrimmedDenseBlock : public Layer<data_t>
        {
        public:
            /// Construct a TrimmedDenseBlock layer.
            explicit TrimmedDenseBlock(index_t axis = -1, const std::string& name = "");

            /// Get the TrimmedDenseBlock axis.
            index_t getAxis() const;

            /// \copydoc Layer::computeOutputDescriptor
            void computeOutputDescriptor() override;

        private:
            index_t axis_;
        };

        /**
         * @brief Class representing the TransitionDown layer in the PhantomNet model
         *
         * @author Andi Braimllari - initial code
         *
         * References:
         * https://arxiv.org/pdf/1811.04602.pdf
         */
        class TransitionDown : public Layer<data_t>
        {
        public:
            /// Construct a TransitionDown layer.
            explicit TransitionDown(index_t axis = -1, const std::string& name = "");

            /// Get the TransitionDown axis.
            index_t getAxis() const;

            /// \copydoc Layer::computeOutputDescriptor
            void computeOutputDescriptor() override;

        private:
            index_t axis_;
        };

        /**
         * @brief Class representing the TransitionUp layer in the PhantomNet model
         *
         * @author Andi Braimllari - initial code
         *
         * References:
         * https://arxiv.org/pdf/1811.04602.pdf
         */
        class TransitionUp : public Layer<data_t>
        {
        public:
            /// Construct a TransitionUp layer.
            explicit TransitionUp(index_t axis = -1, const std::string& name = "");

            /// Get the TransitionUp axis.
            index_t getAxis() const;

            /// \copydoc Layer::computeOutputDescriptor
            void computeOutputDescriptor() override;

        private:
            index_t axis_;
        };

    private:
        template <typename T, typename... Ts>
        void addLayerHelper(T&& first, Ts&&... rest)
        {
            add<T>(std::forward<T>(first));

            if constexpr (sizeof...(rest) > 0) {
                addLayerHelper(std::forward<Ts>(rest)...);
            }
        }

        std::vector<Layer<data_t>*> layers_;
    };
} // namespace elsa::ml
