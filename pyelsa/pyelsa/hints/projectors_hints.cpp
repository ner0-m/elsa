#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

#include "SubsetSampler.h"
#include "SiddonsMethod.h"
#include "JosephsMethod.h"

#ifdef ELSA_CUDA_PROJECTORS
#include "JosephsMethodCUDA.h"
#include "SiddonsMethodCUDA.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class ProjectorsHints : ModuleHints
    {
        struct LoggerProjectors : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerProjectors>(m, "logger_pyelsa_projectors")
                .def_static("setLevel", &LoggerProjectors::setLevel)
                .def_static("enableFileLogging", &LoggerProjectors::enableFileLogging)
                .def_static("flush", &LoggerProjectors::flush);
        }
    };
#endif

    template <typename detectorDescriptor_t, typename data_t>
    class SubsetSamplerHints : public ClassHints<elsa::SubsetSampler<detectorDescriptor_t, data_t>>
    {
    public:
        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            c.def("getProjectorSiddonsMethod",
                  [](SubsetSampler<detectorDescriptor_t, data_t>& self) {
                      return self.template getProjector<SiddonsMethod<data_t>>();
                  });
            c.def("getProjectorJosephsMethod",
                  [](SubsetSampler<detectorDescriptor_t, data_t>& self) {
                      return self.template getProjector<JosephsMethod<data_t>>();
                  });
            c.def("getSubsetProjectorsSiddonsMethod",
                  [](SubsetSampler<detectorDescriptor_t, data_t>& self) {
                      return self.template getSubsetProjectors<SiddonsMethod<data_t>>();
                  });
            c.def("getSubsetProjectorsJosephsMethod",
                  [](SubsetSampler<detectorDescriptor_t, data_t>& self) {
                      return self.template getSubsetProjectors<JosephsMethod<data_t>>();
                  });
#ifdef ELSA_CUDA_PROJECTORS
            c.def("getProjectorSiddonsMethodCUDA",
                  [](SubsetSampler<detectorDescriptor_t, data_t>& self) {
                      return self.template getProjector<SiddonsMethodCUDA<data_t>>();
                  });
            c.def("getProjectorJosephsMethodCUDA",
                  [](SubsetSampler<detectorDescriptor_t, data_t>& self) {
                      return self.template getProjector<JosephsMethodCUDA<data_t>>();
                  });
            c.def("getSubsetProjectorsSiddonsMethodCUDA",
                  [](SubsetSampler<detectorDescriptor_t, data_t>& self) {
                      return self.template getSubsetProjectors<SiddonsMethodCUDA<data_t>>();
                  });
            c.def("getSubsetProjectorsJosephsMethodCUDA",
                  [](SubsetSampler<detectorDescriptor_t, data_t>& self) {
                      return self.template getSubsetProjectors<JosephsMethodCUDA<data_t>>();
                  });
#endif
        }
    };
    template class SubsetSamplerHints<PlanarDetectorDescriptor, float>;
    template class SubsetSamplerHints<PlanarDetectorDescriptor, double>;
    // TODO: figure out why complex subset sampler is not working with python bindings. Usage leads
    // to undefined symbol error on python module import.
    //    template class SubsetSamplerHints<PlanarDetectorDescriptor, std::complex<float>>;
    //    template class SubsetSamplerHints<PlanarDetectorDescriptor, std::complex<double>>;
} // namespace elsa
