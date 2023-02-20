# Create binaries for examples in bin/examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples)

function(add_example example_name example_source)
    add_executable(${example_name} ${example_source})
    target_link_libraries(${example_name} PRIVATE elsa::all)
    target_compile_features(${example_name} PUBLIC cxx_std_17)
    add_dependencies(build-examples ${example_name})
endfunction()

# define a target for all examples
add_custom_target(build-examples)

# build a simple 2D reconstruction
add_example(simple_recon2d simple_recon2d.cpp)

add_subdirectory(misc)
add_subdirectory(phantoms)
add_subdirectory(solver)
add_subdirectory(dataset)
