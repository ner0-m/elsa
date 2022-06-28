# Find Nvidia's Cublas library
#
# The following variables are set if Cublas is found. Cublas_FOUND        - True Cublas_INCLUDE_DIR  - The path to
# cublas.h Cublas_LIBRARY      - The path to the cublas library
#
# This module reads hints about search locations from the following environment variables:
#
# Cublas_ROOT Cublas_ROOT_DIR
#
# as well as in the following subdirectories
#
# include thirdparty ThirdParty external
#
# If Cublas_INCLUDE_DIR or Cublas_LIB are explicitly set, the search falls back to the values in these variables

set(CUBLAS_HINTS ${CUDA_ROOT} $ENV{CUDA_ROOT} $ENV{CUDA_TOOLKIT_ROOT_DIR})
set(CUBLAS_PATHS /usr /usr/local /usr/local/cuda /opt/cuda)

# Finds the include directories
find_path(
    Cublas_INCLUDE_DIRS
    NAMES cublas_v2.h cuda.h
    HINTS ${CUBLAS_HINTS}
    PATH_SUFFIXES include inc include/x86_64 include/x64
    PATHS ${CUBLAS_PATHS}
    DOC "Cublas include header cublas_v2.h"
)
mark_as_advanced(Cublas_INCLUDE_DIRS)

# Finds the libraries
find_library(
    Cuda_LIB
    NAMES cudart
    HINTS ${CUBLAS_HINTS}
    PATH_SUFFIXES
        lib
        lib64
        lib/x86_64
        lib/x64
        lib/x86
        lib/Win32
        lib/import
        lib64/import
    PATHS ${CUBLAS_PATHS}
    DOC "Cuda library"
)
mark_as_advanced(Cuda_LIB)
find_library(
    Cublas_LIB
    NAMES cublas
    HINTS ${CUBLAS_HINTS}
    PATH_SUFFIXES
        lib
        lib64
        lib/x86_64
        lib/x64
        lib/x86
        lib/Win32
        lib/import
        lib64/import
    PATHS ${CUBLAS_PATHS}
    DOC "Cublas library"
)
mark_as_advanced(Cublas_LIB)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cublas DEFAULT_MSG Cublas_LIB Cuda_LIB Cublas_INCLUDE_DIRS)
