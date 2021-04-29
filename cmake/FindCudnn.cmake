# Find Nvidia's Cudnn library
#
# The following variables are set if Cudnn is found.
#   Cudnn_FOUND        - True
#   Cudnn_INCLUDE_DIR  - The path to cudnn.h
#   Cudnn_LIBRARY      - The path to the cudnn library
#
# This module reads hints about search locations from the following environment
# variables:
#
# Cudnn_ROOT Cudnn_ROOT_DIR
#
# as well as in the following subdirectories
#
# include thirdparty ThirdParty external
#
# If
#   Cudnn_INCLUDE_DIR
# or
#   Cudnn_LIB
# are explicitly set, the search falls back to the values in these variables

include(FindPackageHandleStandardArgs)

macro(_cudnn_find_lib)
  find_library(Cudnn_LIB cudnn
               HINTS ENV
                     Cudnn_ROOT
                     ENV
                     Cudnn_ROOT_DIR
                     include
                     thirdparty
                     ThirdParty
                     external
               DOC "Cudnn library files")
endmacro(_cudnn_find_lib)

macro(_cudnn_find_include)
  find_path(Cudnn_INCLUDE_DIR
            NAMES cudnn.h
            HINTS ENV
                  Cudnn_ROOT
                  ENV
                  Cudnn_ROOT_DIR
                  include
                  thirdparty
                  ThirdParty
                  external
            DOC "Cudnn header files")
endmacro(_cudnn_find_include)

if((EXISTS "${Cudnn_INCLUDE_DIR}") AND (EXISTS "${Cudnn_LIB}"))
  find_package_handle_standard_args(Cudnn
                                    DEFAULT_MSG
                                    Cudnn_LIB
                                    Cudnn_INCLUDE_DIR)
  mark_as_advanced(Cudnn_INCLUDE_DIR)
elseif(EXISTS "${Cudnn_INCLUDE_DIR}")
  _cudnn_find_lib()
elseif(EXISTS "${Cudnn_LIB}")
  _cudnn_find_include()
else()
  _cudnn_find_lib()
  _cudnn_find_include()
  find_package_handle_standard_args(Cudnn
                                    DEFAULT_MSG
                                    Cudnn_LIB
                                    Cudnn_INCLUDE_DIR)
  mark_as_advanced(Cudnn_INCLUDE_DIR)
endif()

