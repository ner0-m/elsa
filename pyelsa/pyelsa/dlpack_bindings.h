#pragma once

#include "dlpack/dlpack.h"
/* dlpack mandates a pointer alignment of 256 */
#define DLPACK_ALIGNMENT (256)

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "DataContainer.h"
#include "Descriptors/DataDescriptor.h"
#include "Descriptors/VolumeDescriptor.h"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <limits.h>

namespace detail
{

    namespace py = pybind11;

    template <class data_t>
    void tensor_destructor(DLManagedTensor* self)
    {
        delete[] self->dl_tensor.shape;
        /* strides is NULL, but might be needed in the future */
        delete[] self->dl_tensor.strides;
        elsa::mr::NativeContainer<data_t>* nc =
            static_cast<elsa::mr::NativeContainer<data_t>*>(self->manager_ctx);
        /* this is a bit redundant, deletion should do this as well*/
        nc->release();
        delete nc;
        delete self;
    }

    static void dlpack_capsule_destructor(PyObject* self)
    {
        /* At the risk of editorializing, I would like to say that it is terrible how
            cpython leaks through pybind here. Why have a capsule wrapper at all, if it
            needs to be supplied with a cpython PyCapsule_Destructor? */

        /* used https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/dlpack.c as
         * reference*/

        /* exception free name check */
        if (PyCapsule_IsValid(self, "used_dltensor")) {
            return;
        }

        /* store exception, in case another is raised below*/
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);

        DLManagedTensor* managed =
            reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(self, "dltensor"));
        if (managed == NULL) {
            PyErr_WriteUnraisable(self);
        } else if (managed->deleter) {
            managed->deleter(managed);
        }

        PyErr_Restore(type, value, traceback);
    }

    template <class data_t>
    py::object build_data_container(DLManagedTensor* tensor, py::capsule& capsule)
    {
        static_assert(std::is_trivially_copyable<data_t>::value);

        DLDevice device = tensor->dl_tensor.device;
        elsa::index_t ndim = tensor->dl_tensor.ndim;
        elsa::IndexVector_t shape(ndim);
        for (elsa::index_t i = 0; i < ndim; i++) {
            shape(i) = tensor->dl_tensor.shape[i];
        }

        bool strided_copy = false;
        if (tensor->dl_tensor.strides) {
            /* stride is not in bytes, but array elements */
            elsa::index_t stride = 1;
            for (elsa::index_t i = 0; i < ndim; i++) {
                if (tensor->dl_tensor.strides[ndim - 1 - i] != stride) {
                    strided_copy = true;
                }
                stride *= shape(ndim - 1 - i);
            }
        }

        if (strided_copy) {
            /* TODO */
            PyErr_SetString(PyExc_RuntimeError,
                            "data with non row major densely packed layout not yet supported.");
            return py::none();
        }

        data_t* imported_data_begin =
            static_cast<data_t*>(tensor->dl_tensor.data + tensor->dl_tensor.byte_offset);

        elsa::VolumeDescriptor desc(shape);
        std::unique_ptr<elsa::DataContainer<data_t>> dc;
        if (elsa::mr::storageType() == elsa::mr::StorageType::Host && device.device_type == kDLCPU
            || elsa::mr::storageType() == elsa::mr::StorageType::CUDAManaged
                   && device.device_type == kDLCUDAManaged) {
            /* inform the capsule that the tensor should not be deleted */
            capsule.set_name("used_dltensor");

            dc = std::make_unique<elsa::DataContainer<data_t>>(
                desc, imported_data_begin, elsa::DataContainer<data_t>::ImportStrategy::View,
                [tensor]() { tensor->deleter(tensor); });
        } else {
            auto strategy = elsa::DataContainer<data_t>::ImportStrategy::DeviceCopy;
            if (elsa::mr::storageType() == elsa::mr::StorageType::Host
                && device.device_type == kDLCUDA) {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Unsupported storage type kDLCUDA. elsa is built without CUDA support.");
                return py::none();
            } else if (device.device_type == kDLCPU
                       || elsa::mr::storageType() == elsa::mr::StorageType::Host) {
                strategy = elsa::DataContainer<data_t>::ImportStrategy::HostCopy;
            }
            dc = std::make_unique<elsa::DataContainer<data_t>>(desc, imported_data_begin, strategy,
                                                               []() {});
        }
        return py::cast(dc.release());
    }

    template <class data_t>
    py::object data_container_dlpack(elsa::DataContainer<data_t>& self, py::object stream)
    {
        /* for now, support only stream none, like numpy */
        if (stream != py::none()) {
            PyErr_SetString(PyExc_RuntimeError, "elsa only supports stream=None.");
            return py::none();
        }

        DLDataType dl_type;
        dl_type.bits = CHAR_BIT * sizeof(data_t);
        dl_type.lanes = 1;

        /* Un-comment this if, if C++ 23 is used for this project
        if constexpr (std::is_same<data_t, std::bfloat16_t>::value) {
            dl_type.code = kDLBFloat;
        } */
        if constexpr (std::is_same<data_t, bool>::value) {
            dl_type.code = kDLBool;
        } else if constexpr (std::is_floating_point<data_t>::value) {
            dl_type.code = kDLFloat;
        } else if constexpr (std::is_integral<data_t>::value) {
            if constexpr (std::is_unsigned<data_t>::value) {
                dl_type.code = kDLUInt;
            } else {
                dl_type.code = kDLInt;
            }
        } else if (std::is_same<data_t, elsa::complex<float>>::value
                   || std::is_same<data_t, elsa::complex<double>>::value) {
            dl_type.code = kDLComplex;
        } else {
            PyErr_SetString(PyExc_RuntimeError,
                            "__dlpack__ unsupported for data container of this type.");
            return py::none();
        }

        DLDeviceType dl_device_type;
        switch (elsa::mr::storageType()) {
            case elsa::mr::StorageType::Host:
                dl_device_type = kDLCPU;
                break;
            case elsa::mr::StorageType::CUDAManaged:
                dl_device_type = kDLCUDAManaged;
                break;
            default:
                PyErr_SetString(PyExc_RuntimeError, "Internal error! Unknown storage type.");
                return py::none();
        }

        DLDevice dl_device;
        dl_device.device_type = dl_device_type;
        /* 0 is expected for both managed CUDA memory and regular host memory */
        dl_device.device_id = 0;

        auto& desc = self.getDataDescriptor();
        int32_t ndim = desc.getNumberOfDimensions();
        /* use unique pointers for exception safety */
        std::unique_ptr<int64_t[]> shape = std::make_unique<int64_t[]>(ndim);
        const auto& shape_vector = desc.getNumberOfCoefficientsPerDimension();
        for (size_t i = 0; i < ndim; i++) {
            shape[i] = shape_vector(i);
        }

        std::unique_ptr<DLManagedTensor> managed = std::make_unique<DLManagedTensor>();

        managed->dl_tensor.device = dl_device;
        managed->dl_tensor.dtype = dl_type;

        managed->dl_tensor.ndim = ndim;
        managed->dl_tensor.shape = shape.get();
        /* according to the spec, NULL signifies row-major, densely packed layout */
        managed->dl_tensor.strides = NULL;

        std::unique_ptr<elsa::mr::NativeContainer<data_t>> raw_data =
            std::make_unique<elsa::mr::NativeContainer<data_t>>(self.storage().lock_native());
        managed->manager_ctx = raw_data.get();

        uintptr_t data_ptr = reinterpret_cast<uintptr_t>(raw_data->raw_pointer);
        uintptr_t aligned_data_ptr = data_ptr & ~(DLPACK_ALIGNMENT - 1);
        managed->dl_tensor.data = reinterpret_cast<void*>(aligned_data_ptr);
        managed->dl_tensor.byte_offset = data_ptr - aligned_data_ptr;

        managed->deleter = tensor_destructor<data_t>;

        /* release unique pointers so that lifetime is controlled by
           capsule destructor from now on */
        shape.release();
        managed.release();
        raw_data.release();
        return py::capsule(managed.get(), "dltensor", dlpack_capsule_destructor);
    }

    template <class data_t>
    py::object data_container_from_dlpack(py::object obj)
    {
        py::capsule capsule = obj.attr("__dlpack__")();
        if (capsule == py::none()) {
            return py::none();
        }

        if (std::strcmp(capsule.name(), "dltensor") != 0) {
            PyErr_SetString(PyExc_RuntimeError, "result of __dlpack__ does not contain a tensor.");
            return py::none();
        }

        /* get_pointer never returns null (defined in pytypes.h, nothing in pybind is properly
         * documented)*/
        DLManagedTensor* tensor = capsule.get_pointer<DLManagedTensor>();

        DLDataType dtype = tensor->dl_tensor.dtype;
        if (dtype.lanes != 1) {
            PyErr_SetString(PyExc_RuntimeError, "unsupported data type.");
            return py::none();
        }

        switch (dtype.code) {
            case kDLFloat:
                if (dtype.bits == sizeof(float) * CHAR_BIT) {
                    return build_data_container<float>(tensor, capsule);
                } else if (dtype.bits == sizeof(double) * CHAR_BIT) {
                    return build_data_container<double>(tensor, capsule);
                }
                break;
            case kDLComplex:
                if (dtype.bits == sizeof(elsa::complex<float>) * CHAR_BIT) {
                    return build_data_container<elsa::complex<float>>(tensor, capsule);
                } else if (dtype.bits == sizeof(elsa::complex<double>) * CHAR_BIT) {
                    return build_data_container<elsa::complex<double>>(tensor, capsule);
                }
                break;
#if 0
            case kDLBool:
                if(dtype.bits == sizeof(bool) * CHAR_BIT) {
                    return build_data_container<bool>(tensor, capsule);
                }
                break;
            case kDLInt:
                if(dtype.bits == sizeof(int8_t) * CHAR_BIT) {
                    return build_data_container<int8_t>(tensor, capsule);
                } else if (dtype.bits == sizeof(int16_t) * CHAR_BIT){
                    return build_data_container<int16_t>(tensor, capsule);
                } else if (dtype.bits == sizeof(int32_t) * CHAR_BIT){
                    return build_data_container<int32_t>(tensor, capsule);
                } else if (dtype.bits == sizeof(int64_t) * CHAR_BIT){
                    return build_data_container<int64_t>(tensor, capsule);
                }
                break;
            case kDLUInt:
                if(dtype.bits == sizeof(uint8_t) * CHAR_BIT) {
                    return build_data_container<uint8_t>(tensor, capsule);
                } else if (dtype.bits == sizeof(uint16_t) * CHAR_BIT){
                    return build_data_container<uint16_t>(tensor, capsule);
                } else if (dtype.bits == sizeof(uint32_t) * CHAR_BIT){
                    return build_data_container<uint32_t>(tensor, capsule);
                } else if (dtype.bits == sizeof(uint64_t) * CHAR_BIT){
                    return build_data_container<uint64_t>(tensor, capsule);
                }
                break;
#endif
        }
        PyErr_SetString(PyExc_RuntimeError, "unsupported data type.");
        return py::none();
    }

    template <class data_t>
    void add_data_container_dlpack(py::module& m, py::class_<elsa::DataContainer<data_t>> dc)
    {
        dc.def("__dlpack__", data_container_dlpack<data_t>);

        m.def("from_dlpack", data_container_from_dlpack<data_t>);
    }
} // namespace detail
