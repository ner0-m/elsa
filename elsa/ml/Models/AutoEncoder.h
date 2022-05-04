#pragma once

#include "Model.h"
#include "Softmax.h"
#include "Reshape.h"
#include "Conv.h"
#include "Pooling.h"

namespace elsa::ml
{
    /**
     * @brief Class representing an AutoEncoder model
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t The type of all coefficients used in this model. This parameter is
     * optional and defaults to real_t.
     * @tparam Backend The MlBackend that will be used for inference and training. This
     * parameter is optional and defaults to MlBackend::Auto.
     *
     * References:
     * https://arxiv.org/pdf/2003.05991.pdf
     */
    template <typename data_t = real_t, MlBackend Backend = MlBackend::Dnnl>
    class AutoEncoder : public Model<data_t, Backend>
    {
    public:
        AutoEncoder(VolumeDescriptor inputDescriptor, index_t batchSize);

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
        Input<data_t> _input;

        Conv2D<data_t> _convContr1 = ml::Conv2D<data_t>(16, VolumeDescriptor{{3, 3, 1}},
                                                        ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> _convContr2 = ml::Conv2D<data_t>(32, VolumeDescriptor{{3, 3, 16}},
                                                        ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> _convContr3 = ml::Conv2D<data_t>(64, VolumeDescriptor{{3, 3, 32}},
                                                        ml::Activation::Relu, 1, ml::Padding::Same);

        MaxPooling2D<data_t> _maxPool1 = ml::MaxPooling2D<data_t>();
        MaxPooling2D<data_t> _maxPool2 = ml::MaxPooling2D<data_t>();

        UpSampling2D<data_t> _upsample1 =
            ml::UpSampling2D<data_t>({2, 2}, ml::Interpolation::Bilinear);
        UpSampling2D<data_t> _upsample2 =
            ml::UpSampling2D<data_t>({2, 2}, ml::Interpolation::Bilinear);

        Conv2D<data_t> _convExpan1 = ml::Conv2D<data_t>(32, VolumeDescriptor{{3, 3, 64}},
                                                        ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> _convExpan2 = ml::Conv2D<data_t>(16, VolumeDescriptor{{3, 3, 32}},
                                                        ml::Activation::Relu, 1, ml::Padding::Same);

        Conv2D<data_t> _conv1x1;
    };
} // namespace elsa::ml
