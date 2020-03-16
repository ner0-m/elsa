# Find Quickvec library which implements GPU based vector arithmetic
#
# The following variables are set if Quickvec is found. Quickvec_FOUND - True when the include directory is found.
# Quickvec_INCLUDE_DIR - The path to the headers
#
# This module reads hints about search locations from the following environment variables:
#
# Quickvec_ROOT Quickvec_ROOT_DIR
#
# as well as in the following subdirectories
#
# include thirdparty ThirdParty external
#
# If
#
# Quickvec_INCLUDE_DIR
#
# is explicitly set, the search falls back to the values in these variables

include(FindPackageHandleStandardArgs)

macro(_quickvec_find_include)
  find_path(Quickvec_INCLUDE_DIR
            NAMES quickvec/src/Quickvec.h
            HINTS ENV
                  Quickvec_ROOT
                  ENV
                  Quickvec_ROOT_DIR
                  include
                  thirdparty
                  ThirdParty
                  external
            DOC "Quickvec header files"
            PATH_SUFFIXES quickvec)
endmacro(_quickvec_find_include)

if(NOT EXISTS "${Quickvec_INCLUDE_DIR}")
  _quickvec_find_include()
endif()

find_package_handle_standard_args(Quickvec
                                  DEFAULT_MSG
                                  Quickvec_INCLUDE_DIR)

mark_as_advanced(Quickvec_INCLUDE_DIR)

