# - Try to find libLLVM and libclang-cpp
#
# Defines the variables:
#
#  LLVM_FOUND - LLVM was found of the system
#  LLVM_VERSION - LLVM version
#  LLVM_CXXFLAGS - C++ compiler flags for files that include LLVM headers
#  LLVM_INCLUDE_DIR - Directory containing LLVM headers
#  LLVM_LIB_DIR - Directory containing LLVM libraries
#  LLVM_SHARED_MODE - How the provided components can be collectively linked
#  CLANG_RESOURCE_DIR - Clang resource directory pathname (this is also an INTERNAL CACHE variable)
#

find_program(LLVM_CONFIG
  NAMES llvm-config-10.0 llvm-config100 llvm-config-10
  llvm-config-9.0 llvm-config90 llvm-config-9
  llvm-config-8.0 llvm-config80 llvm-config-8
  llvm-config
  DOC "Path to llvm-config"
)

find_program(CLANG_EXECUTABLE
  NAMES clang-10.0 clang100 clang-10
  clang-9.0 clang90 clang-9
  clang-8.0 clang80 clang-8
  clang
  DOC "Path to clang executable")

if (LLVM_CONFIG)

  if (CLANG_EXECUTABLE)
    execute_process(
      COMMAND ${CLANG_EXECUTABLE} -print-resource-dir
      OUTPUT_VARIABLE CLANG_RESOURCE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CLANG_RESOURCE_DIR ${CLANG_RESOURCE_DIR} CACHE INTERNAL "path to Clang resource directory")
  endif()

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

  execute_process(
    COMMAND ${LLVM_CONFIG} --shared-mode
    OUTPUT_VARIABLE LLVM_SHARED_MODE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  find_library(LLVM_LIBRARY NAMES LLVM PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)

  include(FindPackageHandleStandardArgs)

  if(NOT LLVM_VERSION VERSION_LESS "9.0")
    find_library(CLANG_CPP_LIBRARY NAMES clang-cpp PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_package_handle_standard_args(LLVM VERSION_VAR LLVM_VERSION
                                        REQUIRED_VARS CLANG_RESOURCE_DIR
                                                      LLVM_LIB_DIR
                                                      LLVM_INCLUDE_DIR
                                                      LLVM_LIBRARY
                                                      CLANG_CPP_LIBRARY)
  else()
    find_library(CLANG_AST_LIBRARY NAMES clangAST PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_BASIC_LIBRARY NAMES clangBasic PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_FRONTEND_LIBRARY NAMES clangFrontend PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_TOOLING_LIBRARY NAMES clangTooling PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_DRIVER_LIBRARY NAMES clangDriver PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_LEX_LIBRARY NAMES clangLex PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_PARSE_LIBRARY NAMES clangParse PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_SEMA_LIBRARY NAMES clangSema PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_SERIALIZATION_LIBRARY NAMES clangSerialization PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_ANALYSIS_LIBRARY NAMES clangAnalysis PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CLANG_EDIT_LIBRARY NAMES clangEdit PATHS ${LLVM_LIB_DIR} NO_DEFAULT_PATH)
    find_package_handle_standard_args(LLVM VERSION_VAR LLVM_VERSION
                                         REQUIRED_VARS CLANG_RESOURCE_DIR
                                                       LLVM_LIB_DIR
                                                       LLVM_INCLUDE_DIR
                                                       LLVM_LIBRARY 
                                                       CLANG_AST_LIBRARY 
                                                       CLANG_BASIC_LIBRARY
                                                       CLANG_FRONTEND_LIBRARY
                                                       CLANG_TOOLING_LIBRARY
                                                       CLANG_DRIVER_LIBRARY
                                                       CLANG_LEX_LIBRARY
                                                       CLANG_PARSE_LIBRARY
                                                       CLANG_SEMA_LIBRARY
                                                       CLANG_SERIALIZATION_LIBRARY
                                                       CLANG_ANALYSIS_LIBRARY
                                                       CLANG_EDIT_LIBRARY)
  endif()
else()
  message(WARNING "llvm-config could not be located")
endif()