#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "BaseCircleTrajectoryGenerator.h"
#include "CircleTrajectoryGenerator.h"
#include "CurvedCircleTrajectoryGenerator.h"
#include "EllipseGenerator.h"
#include "NoiseGenerators.h"
#include "Phantoms.h"
#include "SphereTrajectoryGenerator.h"
#include "TrajectoryGenerator.h"

#include "hints/generators_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_generators(py::module& m)
{
    py::class_<elsa::EllipseGenerator<float>> EllipseGeneratorf(m, "EllipseGeneratorf");
    EllipseGeneratorf.def_static(
        "drawFilledEllipse2d",
        (void (*)(elsa::DataContainer<float>&, float, const Eigen::Matrix<long, 2, 1, 0, 2, 1>&,
                  Eigen::Matrix<long, 2, 1, 0, 2, 1>,
                  float))(&elsa::EllipseGenerator<float>::drawFilledEllipse2d),
        py::arg("dc"), py::arg("amplitude"), py::arg("center"), py::arg("sizes"), py::arg("angle"));

    m.attr("EllipseGenerator") = m.attr("EllipseGeneratorf");

    py::class_<elsa::EllipseGenerator<double>> EllipseGeneratord(m, "EllipseGeneratord");
    EllipseGeneratord.def_static(
        "drawFilledEllipse2d",
        (void (*)(elsa::DataContainer<double>&, double, const Eigen::Matrix<long, 2, 1, 0, 2, 1>&,
                  Eigen::Matrix<long, 2, 1, 0, 2, 1>,
                  double))(&elsa::EllipseGenerator<double>::drawFilledEllipse2d),
        py::arg("dc"), py::arg("amplitude"), py::arg("center"), py::arg("sizes"), py::arg("angle"));

    py::module phantoms = m.def_submodule("phantoms", "A set of phantom generators");

    phantoms.def("modifiedSheppLogan",
                 (elsa::DataContainer<float>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                     &elsa::phantoms::modifiedSheppLogan<float>),
                 py::arg("sizes"), py::return_value_policy::move);
    phantoms.def("rectangle",
                 (elsa::DataContainer<float>(*)(
                     Eigen::Matrix<long, -1, 1, 0, -1, 1>, Eigen::Matrix<long, -1, 1, 0, -1, 1>,
                     Eigen::Matrix<long, -1, 1, 0, -1, 1>))(&elsa::phantoms::rectangle<float>),
                 py::arg("volumesize"), py::arg("lower"), py::arg("upper"),
                 py::return_value_policy::move);
    phantoms.def("circular",
                 (elsa::DataContainer<float>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>, float))(
                     &elsa::phantoms::circular<float>),
                 py::arg("volumesize"), py::arg("radius"), py::return_value_policy::move);

    phantoms.def("modifiedSheppLogan",
                 (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                     &elsa::phantoms::modifiedSheppLogan<double>),
                 py::arg("sizes"), py::return_value_policy::move);

    phantoms.def("forbild_head",
                 (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                     &elsa::phantoms::forbild_head<double>),
                 py::arg("sizes"), py::return_value_policy::move);

    phantoms.def("forbild_thorax",
                 (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                     &elsa::phantoms::forbild_thorax<double>),
                 py::arg("sizes"), py::return_value_policy::move);

    phantoms.def("forbild_abdomen",
                 (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                     &elsa::phantoms::forbild_abdomen<double>),
                 py::arg("sizes"), py::return_value_policy::move);

    phantoms.def("rectangle",
                 (elsa::DataContainer<double>(*)(
                     Eigen::Matrix<long, -1, 1, 0, -1, 1>, Eigen::Matrix<long, -1, 1, 0, -1, 1>,
                     Eigen::Matrix<long, -1, 1, 0, -1, 1>))(&elsa::phantoms::rectangle<double>),
                 py::arg("volumesize"), py::arg("lower"), py::arg("upper"),
                 py::return_value_policy::move);
    phantoms.def("circular",
                 (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>, double))(
                     &elsa::phantoms::circular<double>),
                 py::arg("volumesize"), py::arg("radius"), py::return_value_policy::move);

    py::module phantoms_old = phantoms.def_submodule("old", "Old approach for testing.");
    phantoms_old.def("modifiedSheppLogan",
                     (elsa::DataContainer<double>(*)(Eigen::Matrix<long, -1, 1, 0, -1, 1>))(
                         &elsa::phantoms::old::modifiedSheppLogan<double>),
                     py::arg("sizes"), py::return_value_policy::move);

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
    py::class_<elsa::BaseCircleTrajectoryGenerator, elsa::TrajectoryGenerator>
        BaseCircleTrajectoryGenerator(m, "BaseCircleTrajectoryGenerator");
    py::class_<elsa::CircleTrajectoryGenerator, elsa::BaseCircleTrajectoryGenerator>
        CircleTrajectoryGenerator(m, "CircleTrajectoryGenerator");
    CircleTrajectoryGenerator.def_static(
        "createTrajectory",
        (std::unique_ptr<elsa::DetectorDescriptor,
                         std::default_delete<elsa::DetectorDescriptor>>(*)(
            long, const elsa::DataDescriptor&, long, float, float,
            std::optional<elsa::RealVector_t>, std::optional<elsa::RealVector_t>,
            std::optional<elsa::IndexVector_t>,
            std::optional<elsa::RealVector_t>))(&elsa::CircleTrajectoryGenerator::createTrajectory),
        py::arg("numberOfPoses"), py::arg("volumeDescriptor"), py::arg("arcDegrees"),
        py::arg("sourceToCenter"), py::arg("centerToDetector"),
        py::arg("principalPointOffset") = py::none(), py::arg("centerOfRotOffset") = py::none(),
        py::arg("detectorSize") = py::none(), py::arg("detectorSpacing") = py::none());
    py::class_<elsa::CurvedCircleTrajectoryGenerator, elsa::BaseCircleTrajectoryGenerator>
        CurvedCircleTrajectoryGenerator(m, "CurvedCircleTrajectoryGenerator");
    CurvedCircleTrajectoryGenerator.def_static(
        "createTrajectory",
        (std::unique_ptr<elsa::DetectorDescriptor,
                         std::default_delete<elsa::DetectorDescriptor>>(*)(
            long, const elsa::DataDescriptor&, long, float, float, elsa::geometry::Radian,
            std::optional<elsa::RealVector_t>, std::optional<elsa::RealVector_t>,
            std::optional<elsa::IndexVector_t>, std::optional<elsa::RealVector_t>))(
            &elsa::CurvedCircleTrajectoryGenerator::createTrajectory),
        py::arg("numberOfPoses"), py::arg("volumeDescriptor"), py::arg("arcDegrees"),
        py::arg("sourceToCenter"), py::arg("centerToDetector"), py::arg("angle"),
        py::arg("principalPointOffset") = py::none(), py::arg("centerOfRotOffset") = py::none(),
        py::arg("detectorSize") = py::none(), py::arg("detectorSpacing") = py::none());

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
