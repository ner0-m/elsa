# Find Nvidia's Image And Signal Performance Primitives library, NPP
#
# The following variables are set if Cublas is found.
#   Npp_FOUND        - True
#   Npp_INCLUDE_DIR  - The path to cublas.h
#   Npp_LIBRARY      - The path to the cublas library
#
# This module reads hints about search locations from the following environment
# variables:
#
# Npp_ROOT Npp_ROOT_DIR
#
# as well as in the following subdirectories
#
# include thirdparty ThirdParty external
#
# If
#   Npp_INCLUDE_DIR
# or
#   Npp_LIB
# are explicitly set, the search falls back to the values in these variables

include(FindPackageHandleStandardArgs)

set(Npp_HINTS
  ${CUDA_ROOT}
  $ENV{CUDA_ROOT}
  $ENV{CUDA_TOOLKIT_ROOT_DIR}
)
set(Npp_PATHS
  /usr
  /usr/local
  /usr/local/cuda
)

# Finds the include directories
find_path(Npp_INCLUDE_DIRS
  NAMES npp.h nppi.h
  HINTS ${Npp_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${Npp_PATHS}
  DOC "Npp include header cublas_v2.h"
)
mark_as_advanced(Npp_INCLUDE_DIRS)

# Finds the libraries
find_library(Nppc_LIB
  NAMES nppc
  HINTS ${Npp_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${Npp_PATHS}
  DOC "Nppc library"
)
mark_as_advanced(Nppc_LIB)

find_library(Nppig_LIB
  NAMES nppig
  HINTS ${Npp_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${Npp_PATHS}
  DOC "Nppig library"
)
mark_as_advanced(Nppig_LIB)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Npp DEFAULT_MSG Nppc_LIB Nppig_LIB Npp_INCLUDE_DIRS)