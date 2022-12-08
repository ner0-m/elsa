# list all the headers of the module
set(MODULE_HEADERS
    Solver.h
    GradientDescent.h
    PGD.h
    APGD.h
    CGLS.h
    FGM.h
    OGM.h
    ADMM.h
    SQS.h
    OrthogonalMatchingPursuit.h
    LandweberIteration.h
    Landweber.h
    SIRT.h
    AB_GMRES.h
    BA_GMRES.h
)

# list all the code files of the module
set(MODULE_SOURCES
    Solver.cpp
    GradientDescent.cpp
    PGD.cpp
    APGD.cpp
    CGLS.cpp
    FGM.cpp
    OGM.cpp
    SQS.cpp
    OrthogonalMatchingPursuit.cpp
    LandweberIteration.cpp
    Landweber.cpp
    SIRT.cpp
    AB_GMRES.cpp
    BA_GMRES.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging elsa_problems elsa_proximal_operators)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    solvers "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

# We need to compile the DataContainer with the NVCC compiler, if unified memory is used
if(ELSA_CUDA_ENABLED)
    set_source_files_properties(AB_GMRES.cpp PROPERTIES LANGUAGE CUDA)
endif()

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_problems elsa_proximal_operators)
