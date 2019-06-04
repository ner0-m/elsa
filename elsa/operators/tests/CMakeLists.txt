cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# macro for the unit tests
MACRO (ELSA_TEST NAME)
    # create the test executable
    add_executable(test_${NAME} test_${NAME}.cpp test_main.cpp)
    # add catch and the corresponding elsa library
    target_link_libraries(test_${NAME} PRIVATE Catch2::Catch2 ${ELSA_MODULE_TARGET_NAME})
    # enable C++17
    target_compile_features(test_${NAME} PUBLIC cxx_std_17)
    # let Catch discover and register all the test cases
    catch_discover_tests(test_${NAME})
ENDMACRO (ELSA_TEST)


# the actual tests
ELSA_TEST(Identity)
ELSA_TEST(Scaling)
ELSA_TEST(FiniteDifferences)
