#pragma once

#include "Model.h"
#include "Softmax.h"

namespace elsa::ml
{
    /**
     * @brief Class representing the U-Net model
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t The type of all coefficients used in this model. This parameter is optional
     * and defaults to real_t.
     * @tparam Backend The MlBackend that will be used for inference and training. This parameter is
     * optional and defaults to MlBackend::Auto. // TODO Auto or Dnnl?
     *
     * References:
     * https://arxiv.org/pdf/1505.04597.pdf
     */
    template <typename data_t = real_t, MlBackend Backend = MlBackend::Dnnl>
    class UNet : public Model<data_t, Backend>
    {
    public:
        // TODO actually use these constructor inputs
        UNet(index_t inChannels, index_t outChannels) : Model<data_t, Backend>()
        {
            name_ = "UNet";

            convContr1.setInput(&input);
            convContr2.setInput(&convContr1);
            maxPool1.setInput(&convContr2);

            convContr3.setInput(&maxPool1);
            convContr4.setInput(&convContr3);
            maxPool2.setInput(&convContr4);

            convContr5.setInput(&maxPool2);
            convContr6.setInput(&convContr5);
            maxPool3.setInput(&convContr6);

            convContr7.setInput(&maxPool3);
            convContr8.setInput(&convContr7);

            upsample1.setInput(&convContr8);

            convExpan1.setInput(&upsample1);
            convExpan2.setInput(&convExpan1);
            upsample2.setInput(&convExpan2);

            convExpan3.setInput(&upsample2);
            convExpan4.setInput(&convExpan3);
            upsample3.setInput(&convExpan4);

            convExpan5.setInput(&upsample3);
            convExpan6.setInput(&convExpan5);

            conv1x1.setInput(&convExpan6);

            inputs_ = {&input};
            outputs_ = {&conv1x1};

            // save the batch-size this model uses
            batchSize_ = inputs_.front()->getBatchSize();

            // set all input-descriptors by traversing the graph
            setInputDescriptors();
        }

        UNet() : UNet(3, 1){};

        /// make copy constructor deletion explicit
        UNet(const UNet<data_t>&) = delete;

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
        Input<data_t> input = ml::Input(VolumeDescriptor({28, 28, 3}), 1);

        Conv2D<data_t> convContr1 =
            ml::Conv2D<data_t>(64, {3, 3, 3}, ml::Activation::Relu, 1, ml::Padding::Same);
        Conv2D<data_t> convContr2 =
            ml::Conv2D<data_t>(64, {3, 3, 64}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> convContr3 =
            ml::Conv2D<data_t>(128, {3, 3, 64}, ml::Activation::Relu, 1, ml::Padding::Same);
        Conv2D<data_t> convContr4 =
            ml::Conv2D<data_t>(128, {3, 3, 128}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> convContr5 =
            ml::Conv2D<data_t>(256, {3, 3, 128}, ml::Activation::Relu, 1, ml::Padding::Same);
        Conv2D<data_t> convContr6 =
            ml::Conv2D<data_t>(256, {3, 3, 256}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> convContr7 =
            ml::Conv2D<data_t>(512, {3, 3, 256}, ml::Activation::Relu, 1, ml::Padding::Same);
        Conv2D<data_t> convContr8 =
            ml::Conv2D<data_t>(512, {3, 3, 512}, ml::Activation::Relu, 1, ml::Padding::Same);

        // TODO ideally we would only define one as max-pooling does not have trainable parameters
        MaxPooling2D<data_t> maxPool1 = ml::MaxPooling2D();
        MaxPooling2D<data_t> maxPool2 = ml::MaxPooling2D();
        MaxPooling2D<data_t> maxPool3 = ml::MaxPooling2D();

        UpSampling2D<data_t> upsample1 =
            ml::UpSampling2D<data_t>({2, 2}, ml::Interpolation::Bilinear);
        UpSampling2D<data_t> upsample2 =
            ml::UpSampling2D<data_t>({2, 2}, ml::Interpolation::Bilinear);
        UpSampling2D<data_t> upsample3 =
            ml::UpSampling2D<data_t>({2, 2}, ml::Interpolation::Bilinear);

        Conv2D<data_t> convExpan1 =
            ml::Conv2D<data_t>(256, {3, 3, 512}, ml::Activation::Relu, 1, ml::Padding::Same);
        Conv2D<data_t> convExpan2 =
            ml::Conv2D<data_t>(256, {3, 3, 256}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> convExpan3 =
            ml::Conv2D<data_t>(128, {3, 3, 256}, ml::Activation::Relu, 1, ml::Padding::Same);
        Conv2D<data_t> convExpan4 =
            ml::Conv2D<data_t>(128, {3, 3, 128}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> convExpan5 =
            ml::Conv2D<data_t>(64, {3, 3, 128}, ml::Activation::Relu, 1, ml::Padding::Same);
        Conv2D<data_t> convExpan6 =
            ml::Conv2D<data_t>(64, {3, 3, 64}, ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> conv1x1 =
            ml::Conv2D<data_t>(6, {1, 1, 64}, ml::Activation::Relu, 1, ml::Padding::Same);
    };
} // namespace elsa::ml
