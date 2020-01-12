#pragma once

#include <unordered_map>
#include <memory>
#include <iostream>

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

        /// Get this layer's input gradient
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
        struct DnnlMemory {
            /// Memory dimensions
            dnnl::memory::dims dimensions;

            /// Memory descriptor
            dnnl::memory::desc descriptor;

            /// Pointer to memory that was described during layer construction
            std::shared_ptr<dnnl::memory> describedMemory = nullptr;

            /// Pointer to memory that was possibly reordered during execution of
            /// a primitve
            std::shared_ptr<dnnl::memory> effectiveMemory = nullptr;

            /// Dnnl format that for memoryDescriptor
            dnnl::memory::format_tag formatTag;

            /// Flag to indicate whether this memory has been reordered by a
            /// primitive
            bool wasReordered = false;

            /// Flag to indicate whether this memory could be reordered by a
            /// primitive
            bool canBeReordered = false;
        };

        struct PropagationStream {
            /// Vector of primitives this propagation stream consists of
            std::vector<dnnl::primitive> primitives;

            /// Vector of arguments this propagation stream consists of
            std::vector<std::unordered_map<int, dnnl::memory>> arguments;

            /// Flag to indicate whether this propagation stream has been compiled
            bool isCompiled = false;
        };

        /// Construct a DnnlLayer by providing a data descriptors for its input and output
        DnnlLayer(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);

        /// Explicitly deleted copy constructor
        DnnlLayer(const DnnlLayer&) = delete;

        /// Type of all coefficients in this layer, expressed as a Dnnl data-type tag
        static constexpr dnnl::memory::data_type _typeTag = detail::TypeToDnnlTypeTag<data_t>::tag;

        /// Reorders memory from described to effective if memoryDesc differs from memory.get_desc()
        void reorderMemory(const dnnl::memory::desc& memoryDesc, DnnlMemory& memory,
                           PropagationStream& stream);

        /// Compile this layer's backward stream
        virtual void compileBackwardStream() {}

        /// Compile this layer's forward stream
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

        /// This layer's forward propagation stream
        PropagationStream _forwardStream;

        /// This layer's backward propagation stream
        PropagationStream _backwardStream;

        /// This layer's input memory
        DnnlMemory _input;

        /// This layer's input gradient memory
        DnnlMemory _inputGradient;

        /// This layer's output memory
        DnnlMemory _output;

        /// This layer's output gradient memory
        DnnlMemory _outputGradient;

        /// This layer's output DataDescriptor
        std::unique_ptr<DataDescriptor> _outputDescriptor;

        /// This layer's input DataDescriptor
        std::unique_ptr<DataDescriptor> _inputDescriptor;

        /// This layer's Dnnl execution engine
        std::shared_ptr<dnnl::engine> _engine = nullptr;
    };
} // namespace elsa
