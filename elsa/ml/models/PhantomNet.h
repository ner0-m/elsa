#pragma once

namespace elsa
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

        ~PhantomNet()
        {
            for (auto& layer : layers_)
                delete layer;
        }

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
        PhantomNet(const Constraint<data_t>&) = delete;

        /// default destructor
        ~PhantomNet() = default;

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
} // namespace elsa
