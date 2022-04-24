# Create binaries for examples in bin/examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples)

function(add_example example_name example_source)
    add_executable(${example_name} ${example_source})
    target_link_libraries(${example_name} PRIVATE elsa::all)
    target_compile_features(${example_name} PUBLIC cxx_std_17)
    set_quickvec_options(${example_name} ${example_source})
    add_dependencies(build-examples ${example_name})
endfunction()

# define a target for all examples
add_custom_target(build-examples)

# build a simple 2D reconstruction
add_example(simple_recon2d simple_recon2d.cpp)

# build the 2d example program
add_example(example2d example2d.cpp)

# build the 2d limited angle trajectory example program
add_example(limited_angle_example2d limited_angle_example2d.cpp)

# build the 2d example admm program
add_example(example2d_admm example2d_admm.cpp)

# build the shearlet example program
add_example(shearlet_example example2d_shearlet.cpp)

CPMAddPackage(
    NAME argparse
    GITHUB_REPOSITORY p-ranav/argparse
    GIT_TAG v2.2
)

add_example(example_argparse example_argparse.cpp)
target_link_libraries(example_argparse PRIVATE argparse::argparse)

if(ELSA_BUILD_CUDA_PROJECTORS)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
        # build the 3d example program
        add_example(example3d example3d.cpp)

        # build the GPU projector speed test program
        add_example(speed_test speed_test.cpp)
  endif()
endif()
