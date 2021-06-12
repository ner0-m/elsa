cmake_minimum_required(VERSION 3.16)

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

    # compiler specific options
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
            target_link_libraries(pybind11_generator PUBLIC stdc++fs)
        endif()
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        # try to match the stdlib llvm was compiled with
        # if info unavailable, assume it was libstdc++ for Clang and libc++ for AppleClang
        # TODO: generates a dev-warning on initial configuration, but alternative is deprecated
        file(GET_RUNTIME_DEPENDENCIES 
            RESOLVED_DEPENDENCIES_VAR LLVM_DEPS
            EXECUTABLES ${LLVM_CONFIG})
        foreach(LLVM_DEP IN LISTS LLVM_DEPS)
            if (${LLVM_DEP} MATCHES "/libstdc[+][+][.]")
                set(LLVM_CXX_STDLIB_DEP stdc++)
                break()
            elseif(${LLVM_DEP} MATCHES "/libc[+][+][.]")
                set(LLVM_CXX_STDLIB_DEP c++)
                break()
            endif()
        endforeach()
                
        if (LLVM_CXX_STDLIB_DEP)
            target_compile_options(pybind11_generator PUBLIC -stdlib=lib${LLVM_CXX_STDLIB_DEP})
            target_link_options(pybind11_generator PUBLIC -stdlib=lib${LLVM_CXX_STDLIB_DEP}
                                -l${LLVM_CXX_STDLIB_DEP})

            if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
                target_link_libraries(pybind11_generator PUBLIC ${LLVM_CXX_STDLIB_DEP}fs)
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "^[Cc]lang")
            target_compile_options(pybind11_generator PUBLIC -stdlib=libstdc++)
            target_link_options(pybind11_generator PUBLIC -stdlib=libstdc++ -lstdc++)

            if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
                target_link_libraries(pybind11_generator PUBLIC stdc++fs)
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Apple[Cc]lang")
            target_compile_options(pybind11_generator PUBLIC -stdlib=libc++)
            target_link_options(pybind11_generator PUBLIC -stdlib=libc++ -lc++)

            if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
                target_link_libraries(pybind11_generator PUBLIC c++fs)
            endif()
        endif()
    endif()
    
    if (NOT LLVM_VERSION VERSION_LESS 9)
        if (LLVM_SHARED_MODE STREQUAL "shared")
                target_link_libraries(pybind11_generator PUBLIC ${LLVM_LIBRARY} ${CLANG_CPP_LIBRARY})
        elseif (LLVM_SHARED_MODE STREQUAL "static")
                # all of the necessary LLVM components should already be statically linked in clang-cpp
                # when LLVM_SHARED_MODE is static
                target_link_libraries(pybind11_generator PUBLIC ${CLANG_CPP_LIBRARY})
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
                                ${LLVM_LIBRARY})
    endif()

    add_subdirectory(hints)
    if (NOT ELSA_BUILD_PYTHON_BINDINGS)
        # propagate to parent scope
        set(ELSA_BUILD_PYTHON_BINDINGS OFF PARENT_SCORE)
    endif()
endif()
