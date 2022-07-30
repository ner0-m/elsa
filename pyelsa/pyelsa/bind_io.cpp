#include <pybind11/pybind11.h>

#include "EDFHandler.h"
#include "MHDHandler.h"
#include "PGMHandler.h"

#include "hints/io_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_io(py::module& m)
{
    py::class_<elsa::EDF> EDF(m, "EDF");
    EDF.def_static("write",
                   (void (*)(const elsa::DataContainer<float>&,
                             std::basic_ostream<char, std::char_traits<char>>&))(&elsa::EDF::write),
                   py::arg("data"), py::arg("output"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<float>&,
                      std::basic_string<char, std::char_traits<char>, std::allocator<char>>))(
                &elsa::EDF::write),
            py::arg("data"), py::arg("filename"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<double>&,
                      std::basic_ostream<char, std::char_traits<char>>&))(&elsa::EDF::write),
            py::arg("data"), py::arg("output"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<double>&,
                      std::basic_string<char, std::char_traits<char>, std::allocator<char>>))(
                &elsa::EDF::write),
            py::arg("data"), py::arg("filename"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<long>&,
                      std::basic_ostream<char, std::char_traits<char>>&))(&elsa::EDF::write),
            py::arg("data"), py::arg("output"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<long>&,
                      std::basic_string<char, std::char_traits<char>, std::allocator<char>>))(
                &elsa::EDF::write),
            py::arg("data"), py::arg("filename"));

    elsa::EDFHints::addCustomMethods(EDF);

    py::class_<elsa::MHD> MHD(m, "MHD");
    MHD.def_static(
           "write",
           (void (*)(const elsa::DataContainer<float>&,
                     std::basic_string<char, std::char_traits<char>, std::allocator<char>>,
                     std::basic_string<char, std::char_traits<char>, std::allocator<char>>))(
               &elsa::MHD::write),
           py::arg("data"), py::arg("metaFilename"), py::arg("rawFilename"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<double>&,
                      std::basic_string<char, std::char_traits<char>, std::allocator<char>>,
                      std::basic_string<char, std::char_traits<char>, std::allocator<char>>))(
                &elsa::MHD::write),
            py::arg("data"), py::arg("metaFilename"), py::arg("rawFilename"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<long>&,
                      std::basic_string<char, std::char_traits<char>, std::allocator<char>>,
                      std::basic_string<char, std::char_traits<char>, std::allocator<char>>))(
                &elsa::MHD::write),
            py::arg("data"), py::arg("metaFilename"), py::arg("rawFilename"));

    elsa::MHDHints::addCustomMethods(MHD);

    py::class_<elsa::PGM> PGM(m, "PGM");
    PGM.def_static("write",
                   (void (*)(const elsa::DataContainer<float>&,
                             std::basic_ostream<char, std::char_traits<char>>&))(&elsa::PGM::write),
                   py::arg("data"), py::arg("stream"))
        .def_static(
            "write",
            (void (*)(
                const elsa::DataContainer<float>&,
                const std::basic_string<char, std::char_traits<char>, std::allocator<char>>&))(
                &elsa::PGM::write),
            py::arg("data"), py::arg("filename"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<double>&,
                      std::basic_ostream<char, std::char_traits<char>>&))(&elsa::PGM::write),
            py::arg("data"), py::arg("stream"))
        .def_static(
            "write",
            (void (*)(
                const elsa::DataContainer<double>&,
                const std::basic_string<char, std::char_traits<char>, std::allocator<char>>&))(
                &elsa::PGM::write),
            py::arg("data"), py::arg("filename"))
        .def_static(
            "write",
            (void (*)(const elsa::DataContainer<long>&,
                      std::basic_ostream<char, std::char_traits<char>>&))(&elsa::PGM::write),
            py::arg("data"), py::arg("stream"))
        .def_static(
            "write",
            (void (*)(
                const elsa::DataContainer<long>&,
                const std::basic_string<char, std::char_traits<char>, std::allocator<char>>&))(
                &elsa::PGM::write),
            py::arg("data"), py::arg("filename"));

    elsa::IOHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_io, m)
{
    add_definitions_pyelsa_io(m);
}
