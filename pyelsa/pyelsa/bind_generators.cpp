#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "CircleTrajectoryGenerator.h"
#include "EllipseGenerator.h"
#include "NoiseGenerators.h"
#include "PhantomGenerator.h"
#include "SphereTrajectoryGenerator.h"
#include "TrajectoryGenerator.h"

#include "hints/generators_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_generators(py::module& m)
{
    py::class_<elsa::EllipseGenerator<float>> EllipseGeneratorf(m, "EllipseGeneratorf");
    EllipseGeneratorf
        .def_static(
            "drawFilledEllipsoid3d",
            (void (*)(elsa::DataContainer<float>&, float, Eigen::Matrix<long, 3, 1, 0, 3, 1>,
                      Eigen::Matrix<long, 3, 1, 0, 3, 1>, float, float,
                      float))(&elsa::EllipseGenerator<float>::drawFilledEllipsoid3d),
            py::arg("dc"), py::arg("amplitude"), py::arg("center"), py::arg("sizes"),
            py::arg("phi"), py::arg("theta"), py::arg("psi"))
        .def_static(
            "drawFilledEllipse2d",
            (void (*)(elsa::DataContainer<float>&, float, const Eigen::Matrix<long, 2, 1, 0, 2, 1>&,
                      Eigen::Matrix<long, 2, 1, 0, 2, 1>,
                      float))(&elsa::EllipseGenerator<float>::drawFilledEllipse2d),
            py::arg("dc"), py::arg("amplitude"), py::arg("center"), py::arg("sizes"),
            py::arg("angle"));

    m.attr("EllipseGenerator") = m.attr("EllipseGeneratorf");

    py::class_<elsa::EllipseGenerator<double>> EllipseGeneratord(m, "EllipseGeneratord");
    EllipseGeneratord
        .def_static(
            "drawFilledEllipsoid3d",
            (void (*)(elsa::DataContainer<double>&, double, Eigen::Matrix<long, 3, 1, 0, 3, 1>,
                      Eigen::Matrix<long, 3, 1, 0, 3, 1>, double, double,
                      double))(&elsa::EllipseGenerator<double>::drawFilledEllipsoid3d),
            py::arg("dc"), py::arg("amplitude"), py::arg("center"), py::arg("sizes"),
            py::arg("phi"), py::arg("theta"), py::arg("psi"))
        .def_static(
            "drawFilledEllipse2d",
            (void (*)(elsa::DataContainer<double>&, double,
                      const Eigen::Matrix<long, 2, 1, 0, 2, 1>&, Eigen::Matrix<long, 2, 1, 0, 2, 1>,
                      double))(&elsa::EllipseGenerator<double>::drawFilledEllipse2d),
            py::arg("dc"), py::arg("amplitude"), py::arg("center"), py::arg("sizes"),
            py::arg("angle"));

    py::class_<elsa::PhantomGenerator<float>> PhantomGeneratorf(m, "PhantomGeneratorf");
    PhantomGeneratorf
        .def_static("createModifiedSheppLogan",
                    (elsa::DataContainer<float>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                        &elsa::PhantomGenerator<float>::createModifiedSheppLogan),
                    py::arg("sizes"), py::return_value_policy::move)
        .def_static("createRectanglePhantom",
                    (elsa::DataContainer<float>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>,
                                                   Eigen::Matrix<long, -1, 1, 0, -1, 1>,
                                                   Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                        &elsa::PhantomGenerator<float>::createRectanglePhantom),
                    py::arg("volumesize"), py::arg("lower"), py::arg("upper"),
                    py::return_value_policy::move)
        .def_static("createCirclePhantom",
                    (elsa::DataContainer<float>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>, float))(
                        &elsa::PhantomGenerator<float>::createCirclePhantom),
                    py::arg("volumesize"), py::arg("radius"), py::return_value_policy::move);

    m.attr("PhantomGenerator") = m.attr("PhantomGeneratorf");

    py::class_<elsa::PhantomGenerator<double>> PhantomGeneratord(m, "PhantomGeneratord");
    PhantomGeneratord
        .def_static("createModifiedSheppLogan",
                    (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                        &elsa::PhantomGenerator<double>::createModifiedSheppLogan),
                    py::arg("sizes"), py::return_value_policy::move)
        .def_static("createRectanglePhantom",
                    (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>,
                                                    Eigen::Matrix<long, -1, 1, 0, -1, 1>,
                                                    Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                        &elsa::PhantomGenerator<double>::createRectanglePhantom),
                    py::arg("volumesize"), py::arg("lower"), py::arg("upper"),
                    py::return_value_policy::move)
        .def_static("createCirclePhantom",
                    (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>, double))(
                        &elsa::PhantomGenerator<double>::createCirclePhantom),
                    py::arg("volumesize"), py::arg("radius"), py::return_value_policy::move);

    py::class_<elsa::NoNoiseGenerator> NoNoiseGenerator(m, "NoNoiseGenerator");
    NoNoiseGenerator
        .def("operator()",
             (elsa::DataContainer<float>(elsa::NoNoiseGenerator::*)(
                 const elsa::DataContainer<float>&) const)(&elsa::NoNoiseGenerator::operator()),
             py::arg("dc"), py::return_value_policy::move)
        .def("operator()",
             (elsa::DataContainer<double>(elsa::NoNoiseGenerator::*)(
                 const elsa::DataContainer<double>&) const)(&elsa::NoNoiseGenerator::operator()),
             py::arg("dc"), py::return_value_policy::move);

    py::class_<elsa::GaussianNoiseGenerator> GaussianNoiseGenerator(m, "GaussianNoiseGenerator");
    GaussianNoiseGenerator
        .def("operator()",
             (elsa::DataContainer<float>(elsa::GaussianNoiseGenerator::*)(
                 const elsa::DataContainer<float>&)
                  const)(&elsa::GaussianNoiseGenerator::operator()),
             py::arg("dc"), py::return_value_policy::move)
        .def("operator()",
             (elsa::DataContainer<double>(elsa::GaussianNoiseGenerator::*)(
                 const elsa::DataContainer<double>&)
                  const)(&elsa::GaussianNoiseGenerator::operator()),
             py::arg("dc"), py::return_value_policy::move)
        .def(py::init<float, float>(), py::arg("mean"), py::arg("stddev"));

    py::class_<elsa::PoissonNoiseGenerator> PoissonNoiseGenerator(m, "PoissonNoiseGenerator");
    PoissonNoiseGenerator
        .def(
            "operator()",
            (elsa::DataContainer<float>(elsa::PoissonNoiseGenerator::*)(
                const elsa::DataContainer<float>&) const)(&elsa::PoissonNoiseGenerator::operator()),
            py::arg("dc"), py::return_value_policy::move)
        .def("operator()",
             (elsa::DataContainer<double>(elsa::PoissonNoiseGenerator::*)(
                 const elsa::DataContainer<double>&)
                  const)(&elsa::PoissonNoiseGenerator::operator()),
             py::arg("dc"), py::return_value_policy::move)
        .def(py::init<float>(), py::arg("mean"));

    py::class_<elsa::TrajectoryGenerator> TrajectoryGenerator(m, "TrajectoryGenerator");
    py::class_<elsa::CircleTrajectoryGenerator, elsa::TrajectoryGenerator>
        CircleTrajectoryGenerator(m, "CircleTrajectoryGenerator");
    CircleTrajectoryGenerator.def_static(
        "createTrajectory",
        (std::unique_ptr<elsa::DetectorDescriptor,
                         std::default_delete<elsa::DetectorDescriptor>>(*)(
            long, const elsa::DataDescriptor&, long, float, float))(
            &elsa::CircleTrajectoryGenerator::createTrajectory),
        py::arg("numberOfPoses"), py::arg("volumeDescriptor"), py::arg("arcDegrees"),
        py::arg("sourceToCenter"), py::arg("centerToDetector"));

    py::class_<elsa::SphereTrajectoryGenerator, elsa::TrajectoryGenerator>
        SphereTrajectoryGenerator(m, "SphereTrajectoryGenerator");
    SphereTrajectoryGenerator.def_static(
        "createTrajectory",
        (std::unique_ptr<elsa::DetectorDescriptor,
                         std::default_delete<elsa::DetectorDescriptor>>(*)(
            long, const elsa::DataDescriptor&, long, elsa::geometry::SourceToCenterOfRotation,
            elsa::geometry::CenterOfRotationToDetector))(
            &elsa::SphereTrajectoryGenerator::createTrajectory),
        py::arg("numberOfPoses"), py::arg("volumeDescriptor"), py::arg("numberOfCircles"),
        py::arg("sourceToCenter"), py::arg("centerToDetector"));

    elsa::GenratorsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_generators, m)
{
    add_definitions_pyelsa_generators(m);
}