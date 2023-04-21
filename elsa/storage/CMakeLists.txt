# list all the headers of the module
set(MODULE_HEADERS
    # TODO: Add headers from functions/
    ContiguousStorage.h
    TypeTraits.hpp
    reductions/DotProduct.h
    reductions/L0.h
    reductions/L1.h
    reductions/L2.h
    reductions/LInf.h
    reductions/Sum.h
    reductions/Extrema.h
    transforms/InplaceAdd.h
    transforms/InplaceSub.h
    transforms/InplaceMul.h
    transforms/InplaceDiv.h
    transforms/Assign.h
    transforms/Clip.h
    transforms/Cast.h
    transforms/Sub.h
    transforms/Div.h
    transforms/Extrema.h
    transforms/Absolute.h
    transforms/Square.h
    transforms/Sqrt.h
    transforms/Log.h
    transforms/Exp.h
    transforms/Imag.h
    transforms/Real.h
)

# list all the code files of the module
set(MODULE_SOURCES memory_resource/ContiguousMemory.cpp memory_resource/UniversalResource.cpp)

list(APPEND MODULE_PUBLIC_DEPS elsa::Thrust elsa_config)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    storage "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

# If CUDA is enabled, all of the reductions should be compiled with CUDA not the host compiler
if(ELSA_CUDA_ENABLED)
    set_target_properties(elsa_storage PROPERTIES LINKER_LANGUAGE CUDA)
else()
    set_target_properties(elsa_storage PROPERTIES LINKER_LANGUAGE CXX)
endif()

write_module_config(${ELSA_MODULE_NAME})
