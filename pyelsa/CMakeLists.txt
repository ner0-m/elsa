# add_subdirectory(pyelsa/hints)

# pybind11_add_module(_core MODULE pyelsa/main.cpp) target_link_libraries(_core PUBLIC elsa_pybind_hints)
# install(TARGETS _core DESTINATION .)

function(elsa_add_pymodule name)
    pybind11_add_module(pyelsa_${name} MODULE pyelsa/bind_${name}.cpp)
    target_link_libraries(pyelsa_${name} PUBLIC elsa::${name})
    install(TARGETS pyelsa_${name} DESTINATION .)
endfunction()

# Work through them and add them once we decided on an interface
elsa_add_pymodule(core)
elsa_add_pymodule(functionals)
elsa_add_pymodule(generators)
elsa_add_pymodule(io)
elsa_add_pymodule(operators)
elsa_add_pymodule(projectors)
elsa_add_pymodule(proximal_operators)
elsa_add_pymodule(solvers)

if(WITH_CUDA)
    elsa_add_pymodule(projectors_cuda)
endif()
