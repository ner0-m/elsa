# list all the headers of the module
set(MODULE_HEADERS
    elsaDefines.h
    Backtrace.h
    Cloneable.h
    Utilities/Badge.hpp
    Utilities/DataContainerFormatter.hpp
    Utilities/FormatConfig.h
    Utilities/Metrics.hpp
    Utilities/Statistics.hpp
    Utilities/TypeCasts.hpp
    Descriptors/DataDescriptor.h
    Descriptors/DescriptorUtils.h
    Descriptors/VolumeDescriptor.h
    Descriptors/DetectorDescriptor.h
    Descriptors/PlanarDetectorDescriptor.h
    Descriptors/BlockDescriptor.h
    Descriptors/IdenticalBlocksDescriptor.h
    Descriptors/PartitionDescriptor.h
    Descriptors/RandomBlocksDescriptor.h
    DataContainer.h
    DataContainerIterator.h
    Error.h
    Handlers/DataHandler.h
    Handlers/DataHandlerCPU.h
    Handlers/DataHandlerMapCPU.h
    LinearOperator.h
    Expression.h
    ExpressionPredicates.h
    Geometry.h
    StrongTypes.h
)

# list all the code files of the module
set(MODULE_SOURCES
    Backtrace.cpp
    Descriptors/DataDescriptor.cpp
    Descriptors/VolumeDescriptor.cpp
    Descriptors/PlanarDetectorDescriptor.cpp
    Descriptors/RandomBlocksDescriptor.cpp
    Descriptors/DescriptorUtils.cpp
    Descriptors/IdenticalBlocksDescriptor.cpp
    Descriptors/DetectorDescriptor.cpp
    Descriptors/PartitionDescriptor.cpp
    DataContainer.cpp
    Error.cpp
    Handlers/DataHandlerCPU.cpp
    Handlers/DataHandlerMapCPU.cpp
    LinearOperator.cpp
    Geometry.cpp
    StrongTypes.cpp
)

if(${ELSA_BUILD_WITH_QUICKVEC})
    set(MODULE_HEADERS ${MODULE_HEADERS} Handlers/DataHandlerGPU.h Handlers/DataHandlerMapGPU.h)

    set(MODULE_SOURCES ${MODULE_SOURCES} Handlers/DataHandlerGPU.cpp Handlers/DataHandlerMapGPU.cpp)
endif()

list(APPEND MODULE_PUBLIC_DEPS "Eigen3::Eigen")
list(APPEND MODULE_PRIVATE_DEPS)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND MODULE_PUBLIC_DEPS "OpenMP::OpenMP_CXX")
endif()

# link Quickvec for DataHandlerGPU
if(${ELSA_BUILD_WITH_QUICKVEC})
    list(APPEND MODULE_PUBLIC_DEPS "Quickvec::quickvec")
endif()


# use FFTW if available
option(WITH_FFTW "Build elsa using fftw for faster fourier transforms" ON)

# workaround for
# https://github.com/FFTW/fftw3/issues/130
# better would be find_package(fftw3)
# fftw3f: float, fftw3: double, fftw3l: 128bit, _omp: OpenMP
if(WITH_FFTW)
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(FFTW3d IMPORTED_TARGET "fftw3")
        pkg_check_modules(FFTW3f IMPORTED_TARGET "fftw3f")
        if(FFTW3d_FOUND AND FFTW3f_FOUND)
            set(FFTW3_FOUND TRUE)
            list(APPEND MODULE_PUBLIC_DEPS "PkgConfig::FFTW3d" "PkgConfig::FFTW3f")
            # TODO: also add fftw3_omp if supported
        endif()
    endif()
endif()

ADD_ELSA_MODULE(
    core "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

# CMAKE_DL_LIBS is a weird name, so I can't figure out now, how to pass it via argument, so add it later...
target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS})

# Add additional folders for core
target_include_directories(
    ${ELSA_MODULE_TARGET_NAME}
    PUBLIC $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}/Descriptors>
           $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}/Handlers>
           $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}/Utilities>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Descriptors>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Handlers>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Utilities>
)

# Also add the necessary compile definition for quickvec
if(${ELSA_BUILD_WITH_QUICKVEC})
    target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_ENABLE_CUDA_VECTOR)
endif()

if(WITH_FFTW AND FFTW3_FOUND)
    target_compile_definitions("${ELSA_MODULE_TARGET_NAME}" PRIVATE WITH_FFTW)
endif()


if(ELSA_BUILD_PYTHON_BINDINGS)
    set(BINDINGS_SOURCES
        ${MODULE_SOURCES}
        Descriptors/DataDescriptor.h
        Descriptors/DescriptorUtils.h
        Descriptors/VolumeDescriptor.h
        Descriptors/BlockDescriptor.h
        Descriptors/IdenticalBlocksDescriptor.h
        Descriptors/PartitionDescriptor.h
        Descriptors/RandomBlocksDescriptor.h
        Descriptors/DetectorDescriptor.h
        Descriptors/PlanarDetectorDescriptor.h
        Geometry.h
    )

    list(FILTER BINDINGS_SOURCES EXCLUDE REGEX DataHandler.*.cpp)
    list(FILTER BINDINGS_SOURCES EXCLUDE REGEX .*Descriptor.*.cpp)
    list(FILTER BINDINGS_SOURCES EXCLUDE REGEX Geometry.cpp)
    list(FILTER BINDINGS_SOURCES EXCLUDE REGEX StrongTypes.cpp)

    GENERATE_BINDINGS(
        ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
        ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${BINDINGS_SOURCES}
    )
endif()
 
write_module_config(${ELSA_MODULE_NAME})
