# enable ctest and doctest test discovery
include(CTest)
include(doctest)
 
add_custom_target(run-tests-ml_cudnn
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-ml_cudnn
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'ml_cudnn_backend'.")

add_custom_target(build-tests-ml_cudnn) 

ELSA_DOCTEST(CudnnActivation)
ELSA_DOCTEST(CudnnMemory)
ELSA_DOCTEST(CudnnMerging)
ELSA_DOCTEST(CudnnNoop)
ELSA_DOCTEST(CudnnDeconvolution)
ELSA_DOCTEST(CudnnReshape)
