#pragma once

#include "Model.h"
#include "Softmax.h"
#include "Reshape.h"
#include "Conv.h"
#include "Pooling.h"

namespace elsa::ml
{
    /**
     * @brief Class representing the PhantomNet model, a fully convolutional, U-Net like
     * architecture
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t The type of all coefficients used in this model. This parameter is
     * optional and defaults to real_t.
     * @tparam Backend The MlBackend that will be used for inference and training. This
     * parameter is optional and defaults to MlBackend::Auto.
     *
     * References:
     * https://arxiv.org/pdf/1811.04602.pdf
     * https://arxiv.org/pdf/1608.06993.pdf
     */
    template <typename data_t = real_t, MlBackend Backend = MlBackend::Dnnl>
    class PhantomNet : public Model<data_t, Backend>
    {
    public:
        PhantomNet(VolumeDescriptor inputDescriptor, index_t batchSize);

        /// make copy constructor deletion explicit
        PhantomNet(const PhantomNet<data_t>&) = delete;

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
        // TODO should Layer be subclassed here (if so which type, Undefined or a new one?), or
        //  Trainable with no bias an identity activation? or something entirely different?

        /// general TDB consisting of 4 convs.
        // TODO consider ml::Concatenate
        class TrimmedDenseBlock4 : public Layer<data_t>
        {
        public:
            TrimmedDenseBlock4(index_t in_channels, index_t growth_rate);

        private:
            Conv2D<data_t> _conv1;
            Conv2D<data_t> _conv2;
            Conv2D<data_t> _conv3;
            Conv2D<data_t> _conv4;
        };

        /// TDB used in the middle on the architecture, consists of 8 convs.
        // TODO consider ml::Concatenate
        class TrimmedDenseBlock8 : public Layer<data_t>
        {
        public:
            TrimmedDenseBlock8(index_t in_channels, index_t growth_rate);

        private:
            Conv2D<data_t> _conv1;
            Conv2D<data_t> _conv2;
            Conv2D<data_t> _conv3;
            Conv2D<data_t> _conv4;
            Conv2D<data_t> _conv5;
            Conv2D<data_t> _conv6;
            Conv2D<data_t> _conv7;
            Conv2D<data_t> _conv8;
        };

        /// TD
        class TransitionDown : public Layer<data_t>
        {
        public:
            TransitionDown(index_t in_channels, index_t out_channels);

        private:
            Conv2D<data_t> _conv;
            MaxPooling2D<data_t> _maxPool = ml::MaxPooling2D<data_t>();
        };

        /// TU
        class TransitionUp : public Layer<data_t>
        {
        public:
            TransitionUp(index_t in_channels, index_t out_channels);

        private:
            UpSampling2D<data_t> _upsample =
                ml::UpSampling2D<data_t>({2, 2}, ml::Interpolation::Bilinear);
            Conv2D<data_t> _conv;
        };

        Input<data_t> _input;

        Conv2D<data_t> _convStart;

        TrimmedDenseBlock4 _tdb1 = TrimmedDenseBlock4(64, 32);
        TransitionDown _td1 = TransitionDown(128, 128);

        TrimmedDenseBlock4 _tdb2 = TrimmedDenseBlock4(128, 64);
        TransitionDown _td2 = TransitionDown(256, 256);

        TrimmedDenseBlock4 _tdb3 = TrimmedDenseBlock4(256, 128);
        TransitionDown _td3 = TransitionDown(512, 512);

        TrimmedDenseBlock8 _tdb4 = TrimmedDenseBlock8(512, 64);

        TransitionUp _tu1 = TransitionUp(512, 512);
        TrimmedDenseBlock4 _tdb5 = TrimmedDenseBlock4(512 + 512, 64);

        TransitionUp _tu2 = TransitionUp(256, 256);
        TrimmedDenseBlock4 _tdb6 = TrimmedDenseBlock4(256 + 256, 32);

        TransitionUp _tu3 = TransitionUp(128, 128);
        TrimmedDenseBlock4 _tdb7 = TrimmedDenseBlock4(128 + 128, 16);

        Conv2D<data_t> _convEnd;
    };
} // namespace elsa::ml
