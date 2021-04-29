#pragma once

#include "elsaDefines.h"
#include "CudnnCommon.h"
#include "CudnnLayer.h"

#include <npp.h>
#include <nppi.h>

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            /// A layer that reshapes its input without changing the underlying data.
            ///
            /// \author David Tellenbach
            template <typename data_t>
            class CudnnReshape : public CudnnLayer<data_t>
            {
            public:
                /// Construct a CudnnReshape layer.
                CudnnReshape(const VolumeDescriptor& inputDescriptor,
                             const VolumeDescriptor& outputDescriptor);

                /// \copydoc CudnnLayer::compileForwardStream
                void compileForwardStream() override;

                /// \copydoc CudnnLayer::compileBackwardStream
                void compileBackwardStream() override;

            protected:
                using BaseType = CudnnLayer<data_t>;

                /// \copydoc CudnnLayer::input_
                using BaseType::input_;

                /// \copydoc CudnnLayer::inputGradient_
                using BaseType::inputGradient_;

                // \copydoc CudnnLayer::output_
                using BaseType::output_;

                /// \copydoc CudnnLayer::outputGradient_
                using BaseType::outputGradient_;
            };

            /// A layer that flattens its input.
            ///
            /// \author David Tellenbach
            template <typename data_t>
            struct CudnnFlatten : public CudnnReshape<data_t> {
                CudnnFlatten(const VolumeDescriptor& inputDescriptor);
            };

            /// A layer upsamples a given image.
            ///
            /// \author David Tellenbach
            template <typename data_t>
            class CudnnUpsampling : public CudnnLayer<data_t>
            {
            public:
                /// Construct a CudnnUpsampling
                CudnnUpsampling(const VolumeDescriptor& inputDescriptor,
                                const VolumeDescriptor& outputDescriptor,
                                Interpolation interpolation);

                /// \copydoc CudnnLayer::forwardPropagate
                void forwardPropagate() override;

                /// \copydoc CudnnLayer::backwardPropagate
                void backwardPropagate() override;

            private:
                using BaseType = CudnnLayer<data_t>;

                /// \copydoc CudnnLayer::input_
                using BaseType::input_;

                /// \copydoc CudnnLayer::inputGradient_
                using BaseType::inputGradient_;

                // \copydoc CudnnLayer::output_
                using BaseType::output_;

                /// \copydoc CudnnLayer::outputGradient_
                using BaseType::outputGradient_;

                /// Interpolation used during upsampling
                Interpolation interpolation_;

                /// The size of the input as a Nvidia NPP datastructure
                NppiSize inputSize_;

                /// The region-of-interest of the input as a Nvidia NPP datastructure. This is
                /// always equal to the input in our case.
                NppiRect inputROI_;

                /// The size of the output as a Nvidia NPP datastructure
                NppiSize outputSize_;

                /// The region-of-interest of the output as a Nvidia NPP datastructure. This is
                /// always equal to the output-size in our case.
                NppiRect outputROI_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa