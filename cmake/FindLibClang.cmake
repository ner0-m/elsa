find_program(LLVM_CONFIG_FOUND "llvm-config")
if (LLVM_CONFIG_FOUND)
  execute_process(
    COMMAND llvm-config --cxxflags
    OUTPUT_VARIABLE LibClang_Flags)

  string(STRIP ${LibClang_Flags} LibClang_Flags)
  separate_arguments(LibClang_Flags NATIVE_COMMAND ${LibClang_Flags})

  execute_process(
    COMMAND llvm-config --includedir
    OUTPUT_VARIABLE LibClang_INCLUDE_DIR)

  find_library(LibClangCpp_LIBRARY NAMES clang-cpp PATH_SUFFIXES llvm-9/lib llvm-10/lib)
  find_library(LibLLVM_LIBRARY NAMES LLVM PATH_SUFFIXES llvm-9/lib llvm-10/lib)

  set(LibClang_LIBRARIES ${LibClangCpp_LIBRARY} ${LibLLVM_LIBRARY})
  set(LibClang_INCLUDE_DIRS ${LibClang_INCLUDE_DIR})

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(LibClang DEFAULT_MSG LibClang_LIBRARIES LibClang_INCLUDE_DIR)
else()
  message(WARNING "llvm-config couln't be located. Python bindings will not be generated")
endif()