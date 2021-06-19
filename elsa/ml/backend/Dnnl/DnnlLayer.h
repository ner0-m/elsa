#pragma once

#include <unordered_map>
#include <memory>
#include <utility>
#include <string>
#include <vector>

#include "elsaDefines.h"
#include "Common.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "VolumeDescriptor.h"
#include "Logger.h"
#include "TypeCasts.hpp"

#include "dnnl.hpp"

namespace elsa::ml
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

        /// A Dnnl layer
        ///
        /// This class is the base for all Dnnl backend layer's in elsa.
        template <typename data_t>
        class DnnlLayer
        {
        public:
            /// Virtual destructor
            virtual ~DnnlLayer() = default;

            /// Execute this layer's forward primitives on executionStream
            virtual void forwardPropagate(dnnl::stream& executionStream);

            /// Execute this layer's backward primitives on executionStream
            virtual void backwardPropagate(dnnl::stream& executionStream);

            /// Get this layer's input-descriptor at a given index.
            ///
            /// @param index the index of the input-descriptor in this layer's
            /// list of input-descriptors
            /// @return this layer's output-descriptor at a given index
            VolumeDescriptor getInputDescriptor(index_t index = 0) const
            {
                validateVectorIndex(_inputDescriptor, index);
                return *dynamic_unique_ptr_cast<VolumeDescriptor>(
                    _inputDescriptor[asUnsigned(index)]->clone());
            }

            /// Get this layer's output-descriptor
            VolumeDescriptor getOutputDescriptor() const
            {
                assert(_outputDescriptor != nullptr
                       && "Cannot get output-descriptor since it is null");
                return *dynamic_unique_ptr_cast<VolumeDescriptor>(_outputDescriptor->clone());
            }

            /// Set this layer's input at a given index.
            ///
            /// @param input DataContainer containing the input data
            /// @param index Index of the input to set in the list of layer
            /// inputs.
            /// @warning This performs a copy from the DataContainer to Dnnl memory
            /// and is therefore potentially expensive.
            void setInput(const DataContainer<data_t>& input, index_t index = 0);

            /// Set this layer's input memory by passing a pointer to another Dnnl memory
            void setInputMemory(std::shared_ptr<dnnl::memory> input, index_t index = 0);

            /// Set next layer's input memory by passing a pointer to another Dnnl memory
            void setNextInputMemory(std::shared_ptr<dnnl::memory> input);

            /// Set this layer's output-gradient at a given index.
            ///
            /// @param gradient DataContainer containing the gradient data.
            /// @param index Index of the gradient to set in the list of layer
            /// gradients.
            void setOutputGradient(const DataContainer<data_t>& gradient, index_t index = 0);

            /// Set this layer's raw memory storing the gradient of its output
            void setOutputGradientMemory(std::shared_ptr<dnnl::memory> outputGradient,
                                         index_t index = 0);

            /// Set this layer's raw memory storing the gradient of its output
            void setNextOutputGradientMemory(std::shared_ptr<dnnl::memory> outputGradient);

            /// Get the layer's output by copying it into a DataContainer.
            /// If the layer reorders, it gets reordered again.
            ///
            /// \note This method is meant to expose a layer's output. Since
            /// elsa uses whcn as its memory-format, the output gets reshaped
            /// to match this memory-format, regardless of the memory-format
            /// that is used internally.
            ///
            /// @warning This function performs a copy and is therefore potentially
            /// expensive. It should not be used internally to connect network
            /// layers.
            DataContainer<data_t> getOutput() const;

            /// Get a pointer to this layer's dnnl output memory.
            ///
            /// \note In case of reordering primitives the memory returned by
            /// this function can differ from what is expected. In other word,
            /// this function doesn't revert possible memory reordering and
            /// should therefore be used for internal purposes only but not
            /// for final reporting of layer outputs.
            std::shared_ptr<dnnl::memory> getOutputMemory() const;

            /// Get this layer's input gradient
            DataContainer<data_t> getInputGradient(index_t index = 0) const;

            /// Get this layer's input gradient memory
            std::shared_ptr<dnnl::memory> getInputGradientMemory(index_t index = 0);

            /// @returns the number of inputs of this layer
            index_t getNumberOfInputs() const
            {
                return static_cast<index_t>(_inputDescriptor.size());
            }

            /// Set the number of output-gradients of this layer
            void setNumberOfOutputGradients(index_t num)
            {
                _outputGradient =
                    std::vector<DnnlMemory>(num == 0 ? 1 : asUnsigned(num), _outputGradient[0]);
            }

            /// @returns the number of output-gradients of this layer
            index_t getNumberOfOutputGradients() const { return asSigned(_outputGradient.size()); }

            /// Compile this layer, i.e., construct all necessary layer logic based on arguments
            /// defined beforehand.
            ///
            /// @param propagation The kind of propagation this layer should be compiled for
            void compile(PropagationKind propagation = PropagationKind::Forward);

            /// Return a pointer to this layer's execution engine.
            std::shared_ptr<dnnl::engine> getEngine() const;

            /// Set this layer's Dnnl execution engine
            void setEngine(std::shared_ptr<dnnl::engine> engine);

            /// Initialize all parameters of this layer
            virtual void initialize() {}

            /// @returns true if this layer is trainable, false otherwise
            virtual bool isTrainable() const;

            /// @returns true if this layer can merge multiple inputs together,
            /// false otherwise
            virtual bool canMerge() const;

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

            /// A propagation-stream, i.e., a collection of Dnnl primitives
            /// and arguments that can be executed using a Dnnl engine.
            struct PropagationStream {
                /// Vector of primitives this propagation stream consists of
                std::vector<dnnl::primitive> primitives;

                /// Vector of arguments this propagation stream consists of
                std::vector<std::unordered_map<int, dnnl::memory>> arguments;

                /// Flag to indicate whether this propagation stream has been compiled
                bool isCompiled = false;

                // In the case of a debug build we keep a list of primitive names
                std::vector<std::string> names;
            };

#define ELSA_ML_ADD_DNNL_PRIMITIVE(propagationStream, primitive) \
    propagationStream.primitives.push_back(primitive);           \
    propagationStream.names.push_back(#primitive);               \
    Logger::get(this->_name)->trace("Adding Dnnl primitive {}", #primitive)

            std::vector<std::string> getDnnlPrimitiveNames(const PropagationStream& stream)
            {
                return stream.names;
            }

            /// Validate a parameter pack of DnnlMemory
            template <typename... T>
            inline static void validateDnnlMemory([[maybe_unused]] T&&... mem)
            {
#if !defined(NDEBUG)
                (assert(mem != nullptr && "Pointer to Dnnl memory cannot be null"), ...);
                (assert(mem->get_desc().get_size() != 0
                        && "Dnnl memory descriptor cannot be of size 0"),
                 ...);
#endif
            }

            template <typename T>
            inline static void validateVectorIndex([[maybe_unused]] const std::vector<T>& vec,
                                                   [[maybe_unused]] index_t index)
            {
                assert(asUnsigned(index) >= 0 && asUnsigned(index) < vec.size()
                       && "Vector index is out of bounds");
            }

            /// Construct a DnnlLayer by providing a volume-descriptor for its input and output
            DnnlLayer(const VolumeDescriptor& inputDescriptor,
                      const VolumeDescriptor& outputDescriptor, const std::string& name,
                      int allowedNumberOfInputs = 1);

            /// Cosntruct a DnnlLayer by providing a list of volume-descriptors
            /// for its input and a single volume-descriptor for its output
            DnnlLayer(const std::vector<VolumeDescriptor>& inputDescriptor,
                      const VolumeDescriptor& outputDescriptor, const std::string& name,
                      int allowedNumberOfInputs = 1);

            /// Explicitly deleted copy constructor
            DnnlLayer(const DnnlLayer&) = delete;

            /// Type of all coefficients in this layer, expressed as a Dnnl data-type tag
            static constexpr dnnl::memory::data_type _typeTag = TypeToDnnlTypeTag<data_t>::tag;

            /// Reorders memory from described to effective if memory descriptor differs from
            /// primitive description
            void reorderMemory(const dnnl::memory::desc& memoryDesc, DnnlMemory& memory,
                               PropagationStream& stream);

            /// Compile this layer's backward stream
            virtual void compileBackwardStream();

            /// Compile this layer's forward stream
            virtual void compileForwardStream();

            /// If a layer has multiple outputs, it will receive multiple
            /// output-gradients. This functions adds a primitive to the
            /// Dnnl backward-stream that sums up all of these output-gradients.
            void handleMultipleOutputGradients();

            /// Write the content of a DataContainer to Dnnl memory
            ///
            /// \note This performs a copy and is therefore potentially expensive.
            static void writeToDnnlMemory(const DataContainer<data_t>& data, dnnl::memory& memory);

            /// Read the content from Dnnl memory into a DataContainer
            ///
            /// \note This performs a copy and is therefore potentially expensive.
            static void readFromDnnlMemory(DataContainer<data_t>& data, const dnnl::memory& memory);

            /// @returns true if this layer needs to synchronize its Dnnl
            /// execution-stream during a forward-pass, false otherwise.
            ///
            /// This is particularly true for any merging layer.
            virtual bool needsForwardSynchronisation() const { return false; }

            /// @returns true if this layer needs to synchronize its Dnnl
            /// execution-stream during a backward-pass, false otherwise.
            ///
            /// This is particularly true for all layers with multiple outputs.
            virtual bool needsBackwardSynchronisation() const { return _outputGradient.size() > 1; }

            std::string getName() const { return _name; }

            ///  Choose a Dnnl memory format tag, given a VolumeDescriptor.
            ///
            ///  The following format tags are chosen:
            ///
            ///   +-----------+-------+---------+
            ///   | Dimension | Input | Weights |
            ///   +-----------+-------+---------+
            ///   | 2D        | nc    | oi      |
            ///   | 3D        | ncw   | oiw     |
            ///   | 4D        | nchw  | oihw    |
            ///   | 5D        | ncdhw | oidhw   |
            ///   +-----------+-------+---------+
            ///
            /// where each letter has the following meaning:
            ///
            ///  Input case:
            ///  n: Number of batches
            ///  c: Number of input channels
            ///  d: Depth (spatial dimension)
            ///  h: Height (spatial dimension)
            ///  w: Width (spatial dimension)
            ///
            ///  Weights case:
            ///  o: Number if output channels, i.e., number of weights
            ///  i: Number of input channels
            ///  d: Depth (spatial dimension)
            ///  h: Height (spatial dimension)
            ///  w: Width (spatial dimension)
            ///
            /// @param desc DataDescriptor to choose a format type tag.
            /// @param isInput True if the DataDescriptor descripes an input,
            ///                false if it describes weights.
            /// @return Dnnl memory format tag corresponding to the above table.
            static dnnl::memory::format_tag
                dataDescriptorToDnnlMemoryFormatTag(const VolumeDescriptor& desc, bool isInput);

            /// @returns a string representation of a Dnnl memory format-tag
            static std::string dnnlMemoryFormatTagToString(dnnl::memory::format_tag tag);

            /// This layer's forward propagation stream
            PropagationStream _forwardStream;

            /// This layer's backward propagation stream
            PropagationStream _backwardStream;

            /// This layer's input memory
            std::vector<DnnlMemory> _input;

            /// This layer's input gradient memory
            std::vector<DnnlMemory> _inputGradient;

            /// This layer's output memory
            DnnlMemory _output;

            /// Vector with this layer's output-gradients
            std::vector<DnnlMemory> _outputGradient;

            /// This layer's output DataDescriptor
            std::unique_ptr<DataDescriptor> _outputDescriptor;

            /// This layer's input DataDescriptor
            std::vector<std::unique_ptr<DataDescriptor>> _inputDescriptor;

            /// This layer's Dnnl execution engine
            std::shared_ptr<dnnl::engine> _engine = nullptr;

            constexpr static int anyNumberOfInputs = -1;

            int _allowedNumberOfInputs;

            std::string _name;

        private:
            index_t _currentInputMemoryIndex = 0;
            index_t _currentOutputGradientMemoryIndex = 0;
        };
    } // namespace detail

} // namespace elsa::ml
