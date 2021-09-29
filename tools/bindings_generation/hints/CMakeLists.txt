find_package(PythonLibs)
if (PYTHONLIBS_FOUND)
    set(ELSA_PYBIND_HINTS "elsa_pybind_hints")

    # this target is never actually needed or built
    # its only purpose is to force CMake to emit the compile commands for the hints files
    # hints files must have the extension '.cpp' or this will not work
    add_library(${ELSA_PYBIND_HINTS} EXCLUDE_FROM_ALL
                    core_hints.cpp
                    functionals_hints.cpp
                    generators_hints.cpp
                    io_hints.cpp
                    operators_hints.cpp
                    problems_hints.cpp
                    projectors_cuda_hints.cpp
                    projectors_hints.cpp
                    proximity_operators_hints.cpp
                    solvers_hints.cpp
                    tasks_hints.cpp)

    target_include_directories(${ELSA_PYBIND_HINTS} PUBLIC ${PYTHON_INCLUDE_DIRS})
    target_include_directories(${ELSA_PYBIND_HINTS} PUBLIC ${PYBIND11_INCLUDE_DIR})
    target_link_libraries(${ELSA_PYBIND_HINTS} elsa::all)
    target_compile_features(${ELSA_PYBIND_HINTS} PUBLIC cxx_std_17)
else()
    message(WARNING "Python libraries not found. Python bindings will not be generated.")
    set(ELSA_BUILD_PYTHON_BINDINGS OFF PARENT_SCOPE)
endif()