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

    template <typename data_t>
    class DnnlLayer
    {
    public:
        virtual void forwardPropagate(dnnl::stream& executionStream);

        virtual void setInput(const DataContainer<data_t>& input);
        virtual void setInput(const dnnl::memory& input);

        DataContainer<data_t> getOutput();

        /// Compile the layer, i.e., construct all necessary layer logic based on arguments defined
        /// beforehand
        virtual void compile();

        /// Return a pointer to the layer's execution engine
        std::shared_ptr<dnnl::engine> getEngine() const;

    protected:
        /// Construct a DnnlLayer by providing a DataDescriptor for its input and output
        DnnlLayer(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);

        DnnlLayer(const DnnlLayer&) = delete;

        static constexpr dnnl::memory::data_type _typeTag = detail::TypeToDnnlTypeTag<data_t>::tag;

        /// Write the content of a DataContainer to Dnnl memory
        static void writeToDnnlMemory(const DataContainer<data_t>& data, dnnl::memory& memory);

        /// Read the content from Dnnl memory into a DataContainer
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
         * \param desc DataDescriptor to choose a format type tag
         * \param isInput True if the DataDescriptor descripes an input, false if it describes
         *        weights
         */
        static dnnl::memory::format_tag
            dataDescriptorToDnnlMemoryFormatTag(const DataDescriptor& desc, bool isInput);

        /// Flag to indicate whether the execution of the primitive has reordered memory.
        bool _hasReorderedMemory = false;

        /// The layer's Dnnl execution engine
        std::shared_ptr<dnnl::engine> _engine = nullptr;

        dnnl::memory::dims _srcMemoryDimensions;

        /// The layer's source memory descriptor
        dnnl::memory::desc _srcMemoryDescriptor;

        /// The layer's source memory after possible reordering
        dnnl::memory _reorderedSrcMemory;

        /// The layer's destination memory
        std::shared_ptr<dnnl::memory> _srcMemory = nullptr;

        dnnl::memory::dims _dstMemoryDimensions;

        /// The layer's destination memory descriptor
        dnnl::memory::desc _dstMemoryDescriptor;

        /// The layer's destination memory
        dnnl::memory _dstMemory;

        dnnl::memory::format_tag _srcMemoryFormatTag;

        /// Dnnl forward primitive
        std::vector<dnnl::primitive> _forwardPrimitives;

        std::vector<std::unordered_map<int, dnnl::memory>> _forwardArguments;
    };
} // namespace elsa
