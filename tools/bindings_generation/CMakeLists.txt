cmake_minimum_required(VERSION 3.10)

find_package(LibClang)

if (LIBCLANG_FOUND)
    add_executable(pybind11_generator Parser.cpp)
    target_compile_options(pybind11_generator PUBLIC ${LibClang_Flags})
    target_compile_features(pybind11_generator PUBLIC cxx_std_17)
    target_link_libraries(pybind11_generator PUBLIC LLVM clang-cpp stdc++fs)

    file(WRITE ${PROJECT_BINARY_DIR}/elsa/__init__.py "")
else()
    message(WARNING "libclang-dev not found. Python bindings will not be generated.")
endif()
