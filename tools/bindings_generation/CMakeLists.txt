cmake_minimum_required(VERSION 3.10)

find_package(LLVM)

if (NOT LLVM_FOUND)
    message(WARNING "LLVM not found or components missing. Python bindings will not be generated.")
    # Set the to off again, so depending code have it updated 
    set(ELSA_BUILD_PYTHON_BINDINGS OFF PARENT_SCOPE)
else()
    add_executable(pybind11_generator Parser.cpp)
    target_compile_options(pybind11_generator PUBLIC ${LLVM_CXXFLAGS})
    target_compile_features(pybind11_generator PUBLIC cxx_std_17)
    target_include_directories(pybind11_generator PUBLIC ${LLVM_INCLUDE_DIR})

    # linking fails if the generator is built using libc++ 
    # force use of libstdc++ instead when compiling with clang
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(pybind11_generator PUBLIC -stdlib=libstdc++)
        target_link_options(pybind11_generator PUBLIC -stdlib=libstdc++ -lstdc++)
    endif()
    if (NOT LLVM_VERSION VERSION_LESS 9)
        if (LLVM_SHARED_MODE STREQUAL "shared")
                target_link_libraries(pybind11_generator PUBLIC ${LLVM_LIBRARY} ${CLANG_CPP_LIBRARY} stdc++fs)
        elseif (LLVM_SHARED_MODE STREQUAL "static")
                # all of the necessary LLVM components should already be statically linked in clang-cpp
                # when LLVM_SHARED_MODE is static
                target_link_libraries(pybind11_generator PUBLIC ${CLANG_CPP_LIBRARY} stdc++fs)
        endif()
    else()
        target_link_libraries(pybind11_generator PRIVATE 
                                ${CLANG_TOOLING_LIBRARY}
                                ${CLANG_FRONTEND_LIBRARY}
                                ${CLANG_SERIALIZATION_LIBRARY}
                                ${CLANG_PARSE_LIBRARY}
                                ${CLANG_DRIVER_LIBRARY}
                                ${CLANG_SEMA_LIBRARY}
                                ${CLANG_ANALYSIS_LIBRARY}
                                ${CLANG_EDIT_LIBRARY}
                                ${CLANG_AST_LIBRARY}
                                ${CLANG_LEX_LIBRARY}
                                ${CLANG_BASIC_LIBRARY}
                                ${LLVM_LIBRARY}
                                stdc++fs)
    endif()

    add_subdirectory(hints)
    if (NOT ELSA_BUILD_PYTHON_BINDINGS)
        # propagate to parent scope
        set(ELSA_BUILD_PYTHON_BINDINGS OFF PARENT_SCORE)
    endif()
endif()
