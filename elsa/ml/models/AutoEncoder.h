#pragma once

#include "Model.h"
#include "Softmax.h"

namespace elsa::ml
{
    /**
     * @brief Class representing an AutoEncoder model
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t The type of all coefficients used in this model. This parameter is optional
     * and defaults to real_t.
     * @tparam Backend The MlBackend that will be used for inference and training. This parameter is
     * optional and defaults to MlBackend::Auto. // TODO Auto or Dnnl?
     *
     * References:
     * https://arxiv.org/pdf/2003.05991.pdf
     */
    template <typename data_t = real_t, MlBackend Backend = MlBackend::Dnnl>
    class AutoEncoder : public Model<data_t, Backend>
    {
    public:
        // TODO actually use these constructor inputs
        AutoEncoder(index_t inChannels, index_t outChannels) : Model<data_t, Backend>()
        {
            name_ = "AutoEncoder";

            convContr1.setInput(&input);
            maxPool1.setInput(&convContr1);

            convContr2.setInput(&maxPool1);
            maxPool2.setInput(&convContr2);

            convContr3.setInput(&maxPool2);

            upsample1.setInput(&convContr3);
            convExpan1.setInput(&upsample1);

            upsample2.setInput(&convExpan1);
            convExpan2.setInput(&upsample2);

            conv1x1.setInput(&convExpan2);

            inputs_ = {&input};
            outputs_ = {&conv1x1};

            // save the batch-size this model uses
            batchSize_ = inputs_.front()->getBatchSize();

            // set all input-descriptors by traversing the graph
            setInputDescriptors();
        }

        AutoEncoder() : AutoEncoder(3, 1){};

        /// make copy constructor deletion explicit
        AutoEncoder(const AutoEncoder<data_t>&) = delete;

        /// lift the base class variable batchSize_
        using Model<data_t, Backend>::setInputDescriptors;

    protected:
        /// lift the base class variable name_
        using Model<data_t, Backend>::name_;

        /// lift the base class variable inputs_
        using Model<data_t, Backend>::inputs_;

        /// lift the base class variable outputs_
        using Model<data_t, Backend>::outputs_;

        /// lift the base class variable batchSize_
        using Model<data_t, Backend>::batchSize_;

    private:
        // TODO actually use here the constructor inputs
        Input<data_t> input = ml::Input(VolumeDescriptor({28, 28, 1}), 10);

        Conv2D<data_t> convContr1 =
            ml::Conv2D<data_t>(16, {3, 3, 1}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> convContr2 =
            ml::Conv2D<data_t>(32, {3, 3, 16}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> convContr3 =
            ml::Conv2D<data_t>(64, {3, 3, 32}, ml::Activation::Relu, 1, ml::Padding::Same);

        // TODO ideally we would only define one as max-pooling does not have trainable parameters
        MaxPooling2D<data_t> maxPool1 = ml::MaxPooling2D();
        MaxPooling2D<data_t> maxPool2 = ml::MaxPooling2D();

        UpSampling2D<data_t> upsample1 =
            ml::UpSampling2D<data_t>({2, 2}, ml::Interpolation::Bilinear);
        UpSampling2D<data_t> upsample2 =
            ml::UpSampling2D<data_t>({2, 2}, ml::Interpolation::Bilinear);

        Conv2D<data_t> convExpan1 =
            ml::Conv2D<data_t>(32, {3, 3, 64}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> convExpan2 =
            ml::Conv2D<data_t>(16, {3, 3, 32}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> conv1x1 =
            ml::Conv2D<data_t>(1, {1, 1, 16}, ml::Activation::Relu, 1, ml::Padding::Same);
    };
} // namespace elsa::ml
