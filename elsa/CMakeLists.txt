cmake_minimum_required(VERSION 3.9)

# setup macro for component registration
# so you can use find_package(elsa COMPONENT name)
macro(registerComponent name)
    set(ELSA_REGISTERED_COMPONENTS "${ELSA_REGISTERED_COMPONENTS};${name}" PARENT_SCOPE)
endmacro()


# add the elsa modules
add_subdirectory(core)
add_subdirectory(logging)
add_subdirectory(io)
add_subdirectory(operators)
add_subdirectory(functionals)
add_subdirectory(problems)
add_subdirectory(solvers)


# propogate the variable to the parent scope
set(ELSA_REGISTERED_COMPONENTS "${ELSA_REGISTERED_COMPONENTS};" PARENT_SCOPE)
