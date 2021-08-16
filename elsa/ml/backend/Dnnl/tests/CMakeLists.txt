# enable ctest and doctest test discovery
include(CTest)
include(doctest)
 
add_custom_target(run-tests-ml_dnnl
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-ml_dnnl
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'ml_dnnl_backend'.")

add_custom_target(build-tests-ml_dnnl) 

ELSA_DOCTEST(DnnlActivation)
ELSA_DOCTEST(DnnlPooling)
ELSA_DOCTEST(DnnlMerging)
