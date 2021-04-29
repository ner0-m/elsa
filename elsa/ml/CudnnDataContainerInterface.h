#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "CudnnMemory.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        struct CudnnDataContainerInterface {
            static void getCudnnMemoryFromDataContainer(const DataContainer<data_t>& dc,
                                                        CudnnMemory<data_t>* mem)
            {
                switch (dc.getDataHandlerType()) {
                    // If the DataContainer stores its data on the host, we copy it
                    // into the host memory of CudnnMemory
                    case DataHandlerType::CPU:
                    case DataHandlerType::MAP_CPU:
                        mem->allocateHostMemory();
                        std::copy(dc.begin(), dc.end(), mem->hostMemory->getMemoryHandle());
                        break;

                    // If the DataContainer stores its data on the device, we copy it
                    // into the device memory of CudnnMemory
                    case DataHandlerType::GPU:
                    case DataHandlerType::MAP_GPU:
                        mem->allocateDeviceMemory();
                        for (int i = 0; i < mem->deviceMemory->getSize(); ++i) {
                            *(mem->deviceMemory->getMemoryHandle() + i) = dc[i];
                        }
                        break;
                }
            }

            static DataContainer<data_t>
                getDataContainerFromCudnnMemory(const CudnnMemory<data_t>& mem)
            {
                auto& dimensions = mem.getDimensions();
                IndexVector_t dims(dimensions.size());
                for (int i = 0; i < dimensions.size(); ++i) {
                    dims[i] = dimensions[i];
                }

                VolumeDescriptor desc(dims);

                DataContainer dc(desc);
                std::copy(mem.hostMemory->getMemoryHandle(),
                          mem.hostMemory->getMemoryHandle() + mem.hostMemory->getSize(),
                          dc.begin());

                return dc;
            }

            static DataContainer<data_t>
                getDataContainerFromCudnnMemory(const CudnnMemory<data_t>& mem,
                                                const VolumeDescriptor& desc)
            {
                assert(mem.getDimensions()[0] * mem.getDimensions()[1] * mem.getDimensions()[2]
                           * mem.getDimensions()[3]
                       == desc.getNumberOfCoefficients());

                DataContainer dc(desc);
                std::copy(mem.hostMemory->getMemoryHandle(),
                          mem.hostMemory->getMemoryHandle() + mem.hostMemory->getSize(),
                          dc.begin());

                return dc;
            }
        };
    } // namespace detail
} // namespace elsa::ml