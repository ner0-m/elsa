# Create binaries for examples in bin/examples 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples) 
 
# define a target for all examples
add_custom_target(build-examples)

# build the 2d example program
add_executable(example2d example2d.cpp)
target_link_libraries(example2d elsa::all)
target_compile_features(example2d PUBLIC cxx_std_17)
add_dependencies(build-examples example2d)

# build the 2d example admm program
add_executable(example2d_admm example2d_admm.cpp)
target_link_libraries(example2d_admm elsa::all)
target_compile_features(example2d_admm PUBLIC cxx_std_17)
add_dependencies(build-examples example2d_admm)

# build the 2d example shadmm program
add_executable(example2d_shadmm example2d_shadmm.cpp)
target_link_libraries(example2d_shadmm elsa::all)
target_compile_features(example2d_shadmm PUBLIC cxx_std_17)
add_dependencies(build-examples example2d_shadmm)

# build the shearlet example program
add_executable(shearlet_example example2d_shearlet.cpp)
target_link_libraries(shearlet_example elsa::all)
target_compile_features(shearlet_example PUBLIC cxx_std_17)
add_dependencies(build-examples shearlet_example)

# add_executable(exampleMl exampleMl.cpp)
# target_link_libraries(exampleMl elsa::all)
# target_compile_features(exampleMl PUBLIC cxx_std_17)
# add_dependencies(build-examples exampleMl)

# add_executable(example_MNIST example_MNIST.cpp)
# target_link_libraries(example_MNIST elsa::all)
# target_compile_features(example_MNIST PUBLIC cxx_std_17)
# add_dependencies(build-examples example_MNIST)

if(ELSA_BUILD_CUDA_PROJECTORS)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        # build the 3d example program
        add_executable(example3d example3d.cpp)
        target_link_libraries(example3d elsa::all)
        target_compile_features(example3d PUBLIC cxx_std_17)
        add_dependencies(build-examples example3d)

        # build the GPU projector speed test program
        add_executable(speed_test speed_test.cpp)
        target_link_libraries(speed_test elsa::all)
        target_compile_features(speed_test PUBLIC cxx_std_17)
        add_dependencies(build-examples speed_test)
    endif()
endif()
