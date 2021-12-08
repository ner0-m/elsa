# Create binaries for benchmarks in bin/benchmarks
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks) 
 
# add Catch2 only here to still use it in benchmarks 
CPMAddPackage(
     NAME Catch2
     GITHUB_REPOSITORY catchorg/Catch2
     VERSION 2.10.1 # <-- Current, update to: 2.13.1
)

# add the CMake modules for automatic test discovery
set(CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/contrib" ${CMAKE_MODULE_PATH})
 
# enable ctest and Catch test discovery
include(CTest)
include(Catch)

add_custom_target(build-benchmarks)

macro(ELSA_BENCHMARK NAME)
    # Add source to all sources, so we can create a common target to build and run the benchmarks
    SET(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} bench_${NAME}.cpp)

    # create the test executable
    add_executable(bench_${NAME} EXCLUDE_FROM_ALL bench_${NAME}.cpp bench_main.cpp)
    # add catch and the corresponding elsa library
    target_link_libraries(bench_${NAME} PRIVATE Catch2::Catch2 elsa::all)

    if(${ELSA_BUILD_WITH_QUICKVEC})
        # build with nvcc when using quickvec
        set_source_files_properties(bench_${NAME}.cpp bench_main PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(bench_${NAME} PUBLIC ELSA_ENABLE_CUDA_VECTOR)
        target_link_libraries(bench_${NAME} PUBLIC "elsa_quickvec")
        target_compile_features(bench_${NAME} PUBLIC cuda_std_17)
    else()
        # enable C++17
        target_compile_features(bench_${NAME} PUBLIC cxx_std_17)
    endif()

    # include helpers
    target_include_directories(bench_${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/elsa/test_routines/)

    # Add benchmark to build-benchmarks
    add_dependencies(build-benchmarks bench_${NAME})
endmacro(ELSA_BENCHMARK)

ELSA_BENCHMARK(RayGenerationBench)
ELSA_BENCHMARK(Projectors)
ELSA_BENCHMARK(Intersection)
ELSA_BENCHMARK(DataContainerConstruction)
ELSA_BENCHMARK(ExpressionTemplates)
if (ELSA_BUILD_WITH_QUICKVEC)
    ELSA_BENCHMARK(Quickvec)
endif()

# Add a single executable for all benchmarks, as CTest removes a lot of the output
add_executable(bench_all EXCLUDE_FROM_ALL bench_main.cpp ${BENCHMARK_SOURCES})
# add catch and the corresponding elsa library
target_link_libraries(bench_all PRIVATE Catch2::Catch2 elsa::all)
# enable C++17
target_compile_features(bench_all PUBLIC cxx_std_17)
# include helpers
target_include_directories(bench_all PRIVATE ${CMAKE_SOURCE_DIR}/elsa/test_routines/)

# Add the custom target to run all the benchmarks
add_custom_target(run-benchmarks
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bench_all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
        COMMENT "Run benchmarks")
add_dependencies(run-benchmarks bench_all)
