#include "DataContainer.h"
#include "Descriptors/VolumeDescriptor.h"
#include "LinearOperator.h"
#include "DescriptorUtils.h"
#include "Geometry.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <functional>

namespace elsa
{
    template <typename Class>
    struct ClassHints {
    };

    namespace py = pybind11;

    /// wrapper for variadic functions
    template <typename T, std::size_t N, typename = std::make_index_sequence<N>>
    struct invokeBestCommonVariadic;

    template <typename T, std::size_t N, std::size_t... S>
    struct invokeBestCommonVariadic<T, N, std::index_sequence<S...>> {
        static auto exec(const py::args& args)
        {
            if (args.size() == N)
                return elsa::bestCommon(args[S].template cast<T>()...);

            if constexpr (N > 1) {
                return invokeBestCommonVariadic<T, N - 1>::exec(args);
            } else {
                throw(std::logic_error("Unsupported number of variadic arguments"));
            }
        };
    };

    template <typename data_t>
    LinearOperator<data_t> adjointHelper(const LinearOperator<data_t>& op)
    {
        return adjoint(op);
    }

    template <typename data_t>
    LinearOperator<data_t> leafHelper(const LinearOperator<data_t>& op)
    {
        return leaf(op);
    }

    // define global variables and functions in the module hints
    struct ModuleHints {
        static void addCustomFunctions(py::module& m)
        {
            m.def("adjoint", &adjointHelper<float>)
                .def("adjoint", &adjointHelper<double>)
                .def("adjoint", &adjointHelper<std::complex<float>>)
                .def("adjoint", &adjointHelper<std::complex<double>>);

            m.def("leaf", &leafHelper<float>)
                .def("leaf", &leafHelper<double>)
                .def("leaf", &leafHelper<std::complex<float>>)
                .def("leaf", &leafHelper<std::complex<double>>);

            m.def("bestCommon", (std::unique_ptr<DataDescriptor>(*)(
                                    const std::vector<const DataDescriptor*>&))(&bestCommon))
                .def("bestCommon", &invokeBestCommonVariadic<const DataDescriptor&, 10>::exec);
        }
    };

    template <typename data_t, typename type_, typename... options>
    void addOperatorsDc(py::class_<type_, options...>& c)
    {
        c.def(
             "__add__",
             [](const DataContainer<data_t>& self, const DataContainer<data_t>& other) {
                 return DataContainer<data_t>(self + other);
             },
             py::return_value_policy::move)
            .def(
                "__mul__",
                [](const DataContainer<data_t>& self, const DataContainer<data_t>& other) {
                    return DataContainer<data_t>(self * other);
                },
                py::return_value_policy::move)
            .def(
                "__sub__",
                [](const DataContainer<data_t>& self, const DataContainer<data_t>& other) {
                    return DataContainer<data_t>(self - other);
                },
                py::return_value_policy::move)
            .def(
                "__truediv__",
                [](const DataContainer<data_t>& self, const DataContainer<data_t>& other) {
                    return DataContainer<data_t>(self / other);
                },
                py::return_value_policy::move)
            // TODO: make the generator automatically generate this __setitem__ function
            .def("__setitem__", [](elsa::DataContainer<data_t>& dc, elsa::index_t i, data_t value) {
                dc[i] = value;
            });
    }

    template <typename data_t>
    struct DataContainerHints : public ClassHints<elsa::DataContainer<data_t>> {
        constexpr static std::array ignoreMethods = {
            "operator()", "begin", "cbegin", "end", "cend", "rbegin", "crbegin", "rend", "crend"};

        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            addOperatorsDc<data_t>(c);
        }

        template <typename type_, typename... options>
        static void exposeBufferInfo(py::class_<type_, options...>& c)
        {
            c.def(py::init([](py::buffer b) {
                 py::buffer_info info = b.request();

                 if (info.format != py::format_descriptor<data_t>::format())
                     throw std::invalid_argument("Incompatible scalar types");

                 elsa::IndexVector_t coeffsPerDim(info.ndim);

                 ssize_t minStride = info.strides[0];
                 for (std::size_t i = 0; i < static_cast<std::size_t>(info.ndim); i++) {
                     if (info.strides[i] < minStride)
                         minStride = info.strides[i];

                     coeffsPerDim[static_cast<elsa::index_t>(i)] =
                         static_cast<elsa::index_t>(info.shape[i]);
                 }

                 if (static_cast<std::size_t>(minStride) / sizeof(data_t) != 1)
                     throw std::invalid_argument("Cannot convert strided buffer to DataContainer");

                 auto map = Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, 1>>(
                     static_cast<data_t*>(info.ptr), coeffsPerDim.prod());

                 elsa::VolumeDescriptor dd{coeffsPerDim};

                 return std::make_unique<elsa::DataContainer<data_t>>(dd, map);
             })).def_buffer([](elsa::DataContainer<data_t>& m) {
                std::vector<ssize_t> dims, strides;
                auto coeffsPerDim = m.getDataDescriptor().getNumberOfCoefficientsPerDimension();
                ssize_t combined = 1;
                for (int i = 0; i < coeffsPerDim.size(); i++) {
                    dims.push_back(coeffsPerDim[i]);
                    strides.push_back(combined * static_cast<ssize_t>(sizeof(data_t)));
                    combined *= coeffsPerDim[i];
                }
                return py::buffer_info(
                    &m[0], sizeof(data_t), py::format_descriptor<data_t>::format(),
                    m.getDataDescriptor().getNumberOfDimensions(), coeffsPerDim, strides);
            });
        }
    };

    template <typename data_t>
    struct LinearOperatorHints : public ClassHints<elsa::LinearOperator<data_t>> {

        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            c.def(py::self + py::self).def(py::self * py::self);
        }
    };

    template <typename data_t>
    struct DataContainerComplexHints : public ClassHints<elsa::DataContainer<data_t>> {
        constexpr static std::array ignoreMethods = {
            "operator()", "begin", "cbegin", "end", "cend", "rbegin", "crbegin", "rend", "crend"};

        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            addOperatorsDc<data_t>(c);
        }

        // TODO: pybind11 does not provide a default format_descriptor for complex types -> make a
        // custom one for this to work
        // CustomMethod<true, py::buffer_info, elsa::DataContainer<data_t>&> buffer =
        //     [](elsa::DataContainer<data_t>& m) {
        //         std::vector<ssize_t> dims, strides;
        //         auto coeffsPerDim = m.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        //         ssize_t combined = 1;
        //         for (int i = 0; i < coeffsPerDim.size(); i++) {
        //             dims.push_back(coeffsPerDim[i]);
        //             strides.push_back(combined * static_cast<ssize_t>(sizeof(data_t)));
        //             combined *= coeffsPerDim[i];
        //         }
        //         return py::buffer_info(
        //             &m[0],          /* Pointer to buffer */
        //             sizeof(data_t), /* Size of one scalar */
        //             py::format_descriptor<data_t>::format(),
        //             /* Python struct-style format descriptor
        //              */
        //             m.getDataDescriptor().getNumberOfDimensions(), /* Number of dimensions */
        //             coeffsPerDim,                                  /* Buffer dimensions */
        //             strides);
        //     };
    };

    struct GeometryHints : public ClassHints<Geometry> {
        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            // define custom constructors for the constructors accepting an rvalue reference
            c.def(py::init([](geometry::SourceToCenterOfRotation sourceToCenterOfRotation,
                              geometry::CenterOfRotationToDetector centerOfRotationToDetector,
                              geometry::Radian angle, geometry::VolumeData2D volData,
                              geometry::SinogramData2D sinoData,
                              geometry::PrincipalPointOffset offset,
                              geometry::RotationOffset2D centerOfRotOffset) {
                      return std::make_unique<Geometry>(
                          sourceToCenterOfRotation, centerOfRotationToDetector, angle,
                          std::move(volData), std::move(sinoData), offset, centerOfRotOffset);
                  }),
                  py::arg("sourceToCenterOfRotation"), py::arg("centerOfRotationToDetector"),
                  py::arg("angle"), py::arg("volData"), py::arg("sinoData"),
                  py::arg("offset") = geometry::PrincipalPointOffset{0},
                  py::arg("centerOfRotOffset") = geometry::RotationOffset2D{0, 0})
                .def(py::init([](geometry::SourceToCenterOfRotation sourceToCenterOfRotation,
                                 geometry::CenterOfRotationToDetector centerOfRotationToDetector,
                                 geometry::VolumeData3D volData, geometry::SinogramData3D sinoData,
                                 geometry::RotationAngles3D angles,
                                 geometry::PrincipalPointOffset2D offset,
                                 geometry::RotationOffset3D centerOfRotOffset) {
                         return std::make_unique<Geometry>(sourceToCenterOfRotation,
                                                           centerOfRotationToDetector,
                                                           std::move(volData), std::move(sinoData),
                                                           angles, offset, centerOfRotOffset);
                     }),
                     py::arg("sourceToCenterOfRotation"), py::arg("centerOfRotationToDetector"),
                     py::arg("volData"), py::arg("sinoData"), py::arg("angles"),
                     py::arg("offset") = geometry::PrincipalPointOffset2D{0, 0},
                     py::arg("centerOfRotOffset") = geometry::RotationOffset3D{0, 0, 0});
        }
    };

    template <typename TransparentClass>
    struct TransparentClassHints : public ClassHints<TransparentClass> {

        using Vector = std::remove_reference_t<decltype(std::declval<TransparentClass>().get())>;
        using Scalar = decltype(std::declval<Vector>().sum());

        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            c.def(py::init<Vector>())
                .def("__getitem__",
                     (Scalar(TransparentClass::*)(index_t i) const) & TransparentClass::operator[]);
        }
    };

    template struct DataContainerHints<float>;
    template struct DataContainerComplexHints<std::complex<float>>;
    template struct DataContainerHints<double>;
    template struct DataContainerComplexHints<std::complex<double>>;
    template struct DataContainerHints<index_t>;
    template struct LinearOperatorHints<float>;
    template struct LinearOperatorHints<double>;
    template struct LinearOperatorHints<std::complex<float>>;
    template struct LinearOperatorHints<std::complex<double>>;
    template struct TransparentClassHints<geometry::Spacing2D>;
    template struct TransparentClassHints<geometry::OriginShift2D>;
    template struct TransparentClassHints<geometry::Coefficients<2>>;
    template struct TransparentClassHints<geometry::Spacing3D>;
    template struct TransparentClassHints<geometry::OriginShift3D>;
    template struct TransparentClassHints<geometry::Coefficients<3>>;
} // namespace elsa