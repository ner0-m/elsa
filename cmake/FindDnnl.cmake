# Find Intel's Deep Neural Network Library Dnnl
#
# This module currently doesn't support searching for a specific Dnnl version,
# however it will not match Dnnl's predecessor mkl-dnn.
#
# The following variables are set if spdlog is found. Dnnl_FOUND        - True
# when the spdlog include directory is found. Dnnl_INCLUDE_DIR  - The path to
# where the spdlog include files are. Dnnl_LIBRARY      - The path to the Dnnl
# library, i.e., libdnnl.dylib or something similar This module reads hints
# about search locations from the following environment variables:
#
# Dnnl_ROOT Dnnl_ROOT_DIR
#
# as well as in the following subdirectories
#
# include thirdparty ThirdParty external
#
# If
#
# Dnnl_INCLUDE_DIR
#
# or
#
# Dnnl_LIB
#
# are explicitly set, the search falls back to the values in these variables

include(FindPackageHandleStandardArgs)

macro(_dnnl_find_lib)
  find_library(Dnnl_LIB dnnl
               HINTS ENV
                     Dnnl_ROOT
                     ENV
                     Dnnl_ROOT_DIR
                     include
                     thirdparty
                     ThirdParty
                     external
               DOC "Dnnl library files")
endmacro(_dnnl_find_lib)

macro(_dnnl_find_include)
  find_path(Dnnl_INCLUDE_DIR
            NAMES dnnl.hpp
            HINTS ENV
                  Dnnl_ROOT
                  ENV
                  Dnnl_ROOT_DIR
                  include
                  thirdparty
                  ThirdParty
                  external
            DOC "Dnnl header files")
endmacro(_dnnl_find_include)

if((EXISTS "${Dnnl_INCLUDE_DIR}") AND (EXISTS "${Dnnl_LIB}"))
  find_package_handle_standard_args(Dnnl
                                    DEFAULT_MSG
                                    Dnnl_LIB
                                    Dnnl_INCLUDE_DIR)
  mark_as_advanced(Dnnl_INCLUDE_DIR)
elseif(EXISTS "${Dnnl_INCLUDE_DIR}")
  _dnnl_find_lib()
elseif(EXISTS "${Dnnl_LIB}")
  _dnnl_find_include()
else()
  _dnnl_find_lib()
  _dnnl_find_include()
  find_package_handle_standard_args(Dnnl
                                    DEFAULT_MSG
                                    Dnnl_LIB
                                    Dnnl_INCLUDE_DIR)
  mark_as_advanced(Dnnl_INCLUDE_DIR)
endif()

