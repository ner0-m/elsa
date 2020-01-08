#pragma once

#include <unordered_map>
#include <memory>

#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "dnnl.hpp"

namespace elsa
{
    namespace detail
    {
        template <typename data_t>
        struct TypeToDnnlTypeTag {
            static constexpr dnnl::memory::data_type tag = dnnl::memory::data_type::undef;
        };

        template <>
        struct TypeToDnnlTypeTag<float> {
            static constexpr dnnl::memory::data_type tag = dnnl::memory::data_type::f32;
        };
    } // namespace detail

    /**
     * Tag to describe a proagation kind. For some primitives there can
     * be a difference between forward inference and forward training.
     */
    enum PropagationKind { Forward, Backward, Full };

    /**
     * A Dnnl network layer.
     *
     * This clase serves as base class for all Dnnl network layers.
     *
     * \param data_t Type of all coefficients used in the layer
     */
    template <typename data_t>
    class DnnlLayer
    {
    public:
        virtual ~DnnlLayer() = default;

        /// Execute this layer's forward primitives on executionStream
        virtual void forwardPropagate(dnnl::stream& executionStream);

        /// Execute this layer's backward primitives on executionStream
        virtual void backwardPropagate(dnnl::stream& executionStream);

        /**
         * Set this layer's input by passing a DataContainer.
         *
         * \note This performs a copy from the DataContainer to Dnnl memory and is therefore
         * potentially expensive.
         */
        void setInput(const DataContainer<data_t>& input);

        void setOutputGradient(const DataContainer<data_t>& gradient);

        DataContainer<data_t> getInputGradient() const;

        /// Set this layer's input memory by passing a pointer to another Dnnl memory
        virtual void setSourceMemory(std::shared_ptr<dnnl::memory> input);

        /**
         * Get the layer's output by copying it into a DataContainer.
         *
         * If the layer reorders memory, it gets reordered again to match
         * the layer's outputDescriptor.
         */
        DataContainer<data_t> getOutput() const;

        /**
         * Get a pointer to this layer's dnnl output memory.
         *
         * \note In case of reordering primitives the memory returned by this function can differ
         * from what is expected. In other word, this function doesn't revert possible memory
         * reordering and should therefore be used for internal purposes only but not for final
         * reporting of layer outputs.
         */
        std::shared_ptr<dnnl::memory> getOutputMemory();

        /// Compile this layer, i.e., construct all necessary layer logic based on arguments defined
        /// beforehand.
        void compile(PropagationKind propagation = PropagationKind::Forward);

        /// Return a pointer to this layer's execution engine.
        std::shared_ptr<dnnl::engine> getEngine() const;

        /// Set the layer's execution engine
        void setEngine(std::shared_ptr<dnnl::engine> engine);

    protected:
        /// Construct a DnnlLayer by providing a data descriptors for its input and output
        DnnlLayer(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);

        /// Explicitly deleted copy constructor
        DnnlLayer(const DnnlLayer&) = delete;

        /// Type of all coefficients in this layer, expressed as a Dnnl data-type tag
        static constexpr dnnl::memory::data_type _typeTag = detail::TypeToDnnlTypeTag<data_t>::tag;

        virtual void compileBackwardStream() {}

        virtual void compileForwardStream() {}

        /**
         * Write the content of a DataContainer to Dnnl memory
         *
         * \note This performs a copy and is therefore potentially expensive.
         */
        static void writeToDnnlMemory(const DataContainer<data_t>& data, dnnl::memory& memory);

        /**
         * Read the content from Dnnl memory into a DataContainer
         *
         * \note This performs a copy and is therefore potentially expensive.
         */
        static void readFromDnnlMemory(DataContainer<data_t>& data, const dnnl::memory& memory);

        /**
         *  Choose a Dnnl memory format tag from a given DataDescriptor.
         *
         *  The following format tags are chosen:
         *
         *   +-----------+-------+---------+
         *   | Dimension | Input | Weights |
         *   +-----------+-------+---------+
         *   | 2D        | nc    | oi      |
         *   | 3D        | ncw   | oiw     |
         *   | 4D        | nchw  | oihw    |
         *   | 5D        | ncdhw | oidhw   |
         *   +-----------+-------+---------+
         *
         * where each letter has the following meaning:
         *
         *  Input case:
         *  n: Number of batches
         *  c: Number of input channels
         *  d: Depth (spatial dimension)
         *  h: Height (spatial dimension)
         *  w: Width (spatial dimension)
         *
         *  Weights case:
         *  o: Number if output channels, i.e., number of weights
         *  i: Number of input channels
         *  d: Depth (spatial dimension)
         *  h: Height (spatial dimension)
         *  w: Width (spatial dimension)
         *
         * \param desc DataDescriptor to choose a format type tag
         * \param isInput True if the DataDescriptor descripes an input, false if it describes
         *        weights
         * \return Dnnl memory format tag corresponding to the above table
         */
        static dnnl::memory::format_tag
            dataDescriptorToDnnlMemoryFormatTag(const DataDescriptor& desc, bool isInput);

        /// Flag to indicate whether the execution of the primitive has reordered memory.
        bool _hasReorderedMemory = false;

        /// Flag to indicate whether the execution of a primitve may reoder memory for better
        /// performance
        bool _mayReorderMemory = false;

        /// The layer's Dnnl execution engine
        std::shared_ptr<dnnl::engine> _engine = nullptr;

        /// Dimensions of this layer's source memory
        dnnl::memory::dims _srcMemoryDimensions;

        /// The layer's source memory descriptor
        dnnl::memory::desc _srcMemoryDescriptor;

        /// The layer's source memory after possible reordering
        dnnl::memory _reorderedSrcMemory;

        /// The layer's source memory
        std::shared_ptr<dnnl::memory> _srcMemory = nullptr;

        dnnl::memory::dims _dstMemoryDimensions;

        /// The layer's destination memory descriptor
        dnnl::memory::desc _dstMemoryDescriptor;

        /// The layer's destination memory
        std::shared_ptr<dnnl::memory> _dstMemory;

        /// Format that of Dnnl source memory
        dnnl::memory::format_tag _srcMemoryFormatTag;

        /// Format that of Dnnl destination memory
        dnnl::memory::format_tag _dstMemoryFormatTag;

        /// Dnnl forward primitives
        std::vector<dnnl::primitive> _forwardPrimitives;

        /// Dnnl backward primitives
        std::vector<dnnl::primitive> _backwardPrimitives;

        std::unique_ptr<DataDescriptor> _outputDescriptor;

        std::unique_ptr<DataDescriptor> _inputDescriptor;

        /// Dnnl forward arguments, i.e., arguments for executing primitives
        std::vector<std::unordered_map<int, dnnl::memory>> _forwardArguments;

        /// Dnnl backward arguments, i.e., arguments for executing primitives
        std::vector<std::unordered_map<int, dnnl::memory>> _backwardArguments;

        dnnl::memory::desc _gradientDstMemoryDescriptor;
        std::shared_ptr<dnnl::memory> _gradientDstMemory;
        dnnl::memory _reorderedGradientDstMemory;

        dnnl::memory::desc _gradientSrcMemoryDescriptor;
        dnnl::memory _gradientSrcMemory;
    };
} // namespace elsa
