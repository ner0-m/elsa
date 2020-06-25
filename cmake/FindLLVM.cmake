# - Try to find libLLVM and libclang-cpp
#
# Defines the variables:
#
#  LLVM_FOUND - LLVM was found of the system
#  LLVM_VERSION - LLVM version
#  LLVM_CXXFLAGS - C++ compiler flags for files that include LLVM headers
#  LLVM_INCLUDE_DIR - Directory containing LLVM headers
#  LLVM_LIB_DIR - Directory containing LLVM libraries
#

find_program(LLVM_CONFIG
  NAMES llvm-config-10.0 llvm-config100 llvm-config-10
  llvm-config-9.0 llvm-config90 llvm-config-9
  llvm-config
  DOC "Path to llvm-config"
)

if (LLVM_CONFIG)
  execute_process(
    COMMAND ${LLVM_CONFIG} --version
    OUTPUT_VARIABLE LLVM_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
    COMMAND ${LLVM_CONFIG} --cxxflags
    OUTPUT_VARIABLE LLVM_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  separate_arguments(LLVM_CXXFLAGS NATIVE_COMMAND ${LLVM_CXXFLAGS})

  execute_process(
    COMMAND ${LLVM_CONFIG} --includedir
    OUTPUT_VARIABLE LLVM_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  
  
  execute_process(
    COMMAND ${LLVM_CONFIG} --libdir
    OUTPUT_VARIABLE LLVM_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  find_library(CLANG_CPP_LIBRARY NAMES clang-cpp PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
  find_library(LLVM_LIBRARY NAMES LLVM PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(LLVM DEFAULT_MSG LLVM_LIB_DIR LLVM_INCLUDE_DIR LLVM_LIBRARY CLANG_CPP_LIBRARY)
else()
  message(WARNING "llvm-config could not be located")
endif()