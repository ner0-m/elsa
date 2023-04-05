# list all the headers of the module
set(MODULE_HEADERS Geometry.h DetectorDescriptor.h PlanarDetectorDescriptor.h CurvedDetectorDescriptor.h)

# list all the code files of the module
set(MODULE_SOURCES Geometry.cpp CurvedDetectorDescriptor.cpp DetectorDescriptor.cpp PlanarDetectorDescriptor.cpp)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    xray_common "${MODULE_HEADERS}" "${MODULE_SOURCES}" PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging)
