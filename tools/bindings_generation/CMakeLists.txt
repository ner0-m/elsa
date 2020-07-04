cmake_minimum_required(VERSION 3.10)

find_package(LLVM)

if (NOT LLVM_FOUND)
    message(WARNING "LLVM not found or components missing. Python bindings will not be generated.")
elseif (LLVM_VERSION VERSION_LESS 9.0.0)
    message(WARNING "LLVM version ${LLVM_VERSION} found in directory ${LLVM_LIB_DIR}, "
                    "but at least 9.0.0 is required. Python bindings will not be generated.")
else()
    add_executable(pybind11_generator Parser.cpp)
    target_compile_options(pybind11_generator PUBLIC ${LLVM_CXXFLAGS})
    target_compile_features(pybind11_generator PUBLIC cxx_std_17)
    target_link_directories(pybind11_generator PUBLIC ${LLVM_LIB_DIR})
    target_link_libraries(pybind11_generator LLVM clang-cpp stdc++fs)

    file(WRITE ${PROJECT_BINARY_DIR}/elsa/__init__.py "")
endif()
