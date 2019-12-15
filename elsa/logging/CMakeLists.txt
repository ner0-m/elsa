cmake_minimum_required(VERSION 3.9)

# list all the headers of the module
set(MODULE_HEADERS
        Logger.h
        LogGuard.h
        Timer.h)

# list all the code files of the module
set(MODULE_SOURCES
        Logger.cpp
        LogGuard.cpp
        Timer.cpp)


# setup library see elsa/CMakeLists.txt (it's a macro, so we can set variables here)
ELSA_LIBRARY_SETUP(logging)

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC spdlog::spdlog_header_only)
