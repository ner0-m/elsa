# list all the headers of the module
set(MODULE_HEADERS
    elsaDefines.h
    Backtrace.h
    Cloneable.h
    Utilities/Assertions.h
    Utilities/Badge.hpp
    Utilities/Bessel.h
    Utilities/CartesianIndices.h
    Utilities/DataContainerFormatter.hpp
    Utilities/FormatConfig.h
    Utilities/Math.hpp
    Utilities/Statistics.hpp
    Utilities/TypeCasts.hpp
    Descriptors/DataDescriptor.h
    Descriptors/DescriptorUtils.h
    Descriptors/VolumeDescriptor.h
    Descriptors/DetectorDescriptor.h
    Descriptors/CurvedDetectorDescriptor.h
    Descriptors/PlanarDetectorDescriptor.h
    Descriptors/BlockDescriptor.h
    Descriptors/IdenticalBlocksDescriptor.h
    Descriptors/PartitionDescriptor.h
    Descriptors/RandomBlocksDescriptor.h
    DataContainer.h
    DataContainerIterator.h
    Error.h
    LinearOperator.h
    ExpressionPredicates.h
    Geometry.h
    StrongTypes.h
)

# list all the code files of the module
set(MODULE_SOURCES
    elsaDefines.cpp
    Backtrace.cpp
    Utilities/Bessel.cpp
    Utilities/CartesianIndices.cpp
    Utilities/Assertions.cpp
    Descriptors/DataDescriptor.cpp
    Descriptors/VolumeDescriptor.cpp
    Descriptors/CurvedDetectorDescriptor.cpp
    Descriptors/PlanarDetectorDescriptor.cpp
    Descriptors/RandomBlocksDescriptor.cpp
    Descriptors/DescriptorUtils.cpp
    Descriptors/IdenticalBlocksDescriptor.cpp
    Descriptors/DetectorDescriptor.cpp
    Descriptors/PartitionDescriptor.cpp
    DataContainer.cpp
    Error.cpp
    LinearOperator.cpp
    Geometry.cpp
    StrongTypes.cpp
)

list(APPEND MODULE_PUBLIC_DEPS "Eigen3::Eigen" elsa_logging elsa_storage)
list(APPEND MODULE_PRIVATE_DEPS)

# use OpenMP if available
if(OpenMP_CXX_FOUND)
    list(APPEND MODULE_PUBLIC_DEPS "OpenMP::OpenMP_CXX")
endif()

# use FFTW if available
option(WITH_FFTW "Build elsa using fftw for faster fourier transforms" ON)

# workaround for https://github.com/FFTW/fftw3/issues/130 better would be find_package(fftw3) fftw3f: float, fftw3:
# double, fftw3l: 128bit, _omp: OpenMP
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
           $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}/Utilities>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Descriptors>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Utilities>
)

if(WITH_FFTW AND FFTW3_FOUND)
    target_compile_definitions("${ELSA_MODULE_TARGET_NAME}" PRIVATE WITH_FFTW)
endif()

# We need to compile the DataContainer with the NVCC compiler, if unified memory is used
if(ELSA_CUDA_ENABLED)
    set_source_files_properties(DataContainer.cpp PROPERTIES LANGUAGE CUDA)
endif()

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_logging)
