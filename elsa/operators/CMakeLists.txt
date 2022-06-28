# list all the headers of the module
set(MODULE_HEADERS
    Identity.h
    Scaling.h
    FiniteDifferences.h
    FourierTransform.h
    BlockLinearOperator.h
    Dictionary.h
    ShearletTransform.h
)

# list all the code files of the module
set(MODULE_SOURCES
    Identity.cpp
    Scaling.cpp
    FiniteDifferences.cpp
    FourierTransform.cpp
    BlockLinearOperator.cpp
    Dictionary.cpp
    ShearletTransform.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PRIVATE_DEPS)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND MODULE_PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

ADD_ELSA_MODULE(
    operators "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging)
