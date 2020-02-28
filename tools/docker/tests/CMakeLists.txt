cmake_minimum_required(VERSION 3.10)

# set the project name
project(Test)

# add the executable
add_executable(Test test.cpp)
set_target_properties(Test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
