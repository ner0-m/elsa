# Create binaries for examples in bin/examples 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples)

# define a target for all examples
add_custom_target(build-examples)

# build the 2d example program
add_executable(example2d example2d.cpp)
target_link_libraries(example2d PRIVATE elsa::all)
target_compile_features(example2d PUBLIC cxx_std_17)
set_quickvec_options(example2d example2d.cpp)
add_dependencies(build-examples example2d)

# build a temporary 2d example program
add_executable(example2d_ctm example2d_ctm.cpp)
target_link_libraries(example2d_ctm PRIVATE elsa::all)
target_compile_features(example2d_ctm PUBLIC cxx_std_17)
set_quickvec_options(example2d_ctm example2d_ctm.cpp)
add_dependencies(build-examples example2d_ctm)

# build the 2d limited angle trajectory example program
add_executable(limited_angle_example2d limited_angle_example2d.cpp limited_angle_example2d.cpp)
target_link_libraries(limited_angle_example2d PRIVATE elsa::all)
target_compile_features(limited_angle_example2d PUBLIC cxx_std_17)
set_quickvec_options(limited_angle_example2d limited_angle_example2d.cpp limited_angle_example2d.cpp)
add_dependencies(build-examples limited_angle_example2d)

# build the 2d example admm program
add_executable(example2d_admm example2d_admm.cpp)
target_link_libraries(example2d_admm PRIVATE elsa::all)
target_compile_features(example2d_admm PUBLIC cxx_std_17)
set_quickvec_options(example2d_admm example2d_admm.cpp)
add_dependencies(build-examples example2d_admm)

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
        target_link_libraries(example3d PRIVATE elsa::all)
        target_compile_features(example3d PUBLIC cxx_std_17)
        set_quickvec_options(example3d example3d.cpp)
        add_dependencies(build-examples example3d)

        # build the GPU projector speed test program
        add_executable(speed_test speed_test.cpp)
        target_link_libraries(speed_test PRIVATE elsa::all)
        target_compile_features(speed_test PUBLIC cxx_std_17)
        set_quickvec_options(speed_test speed_test.cpp)
        add_dependencies(build-examples speed_test)
    endif()
endif()
