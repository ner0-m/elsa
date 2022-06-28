# * Try to find libLLVM and libclang-cpp
#
# Defines the variables:
#
# LLVM_FOUND - LLVM was found on the system LLVM_VERSION - LLVM version LLVM_CXXFLAGS - C++ compiler flags for files
# that include LLVM headers LLVM_INCLUDE_DIR - Directory containing LLVM headers LLVM_LIB_DIR - Directory containing
# LLVM libraries LLVM_CXX_STDLIB_DEP - the c++ stdlib which LLVM depends on IS_LLVM_STATICALLY_LINKED - are LLVM (or its
# components) statically linked into libTooling CLANG_RESOURCE_DIR - Clang resource directory pathname (this is also an
# INTERNAL CACHE variable) LIBCXX_INCLUDE_DIR - path to the libc++ headers (this is also an INTERNAL CACHE variable)
#

# set the CLANG_VERSION variables from the output emitted by clang --version
function(_set_clang_version CLANG_EXECUTABLE)
    execute_process(COMMAND ${CLANG_EXECUTABLE} --version OUTPUT_VARIABLE CLANG_OUT OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX MATCH "clang version ([0-9]+)\.([0-9]+)\.([0-9]+)" REGEX_MATCH ${CLANG_OUT})
    if(REGEX_MATCH STREQUAL "")
        message(WARNING "Could not determine clang version")
    else()
        set(CLANG_VERSION_MAJOR ${CMAKE_MATCH_1} PARENT_SCOPE)
        set(CLANG_VERSION_MINOR ${CMAKE_MATCH_2} PARENT_SCOPE)
        set(CLANG_VERSION_PATCH ${CMAKE_MATCH_3} PARENT_SCOPE)
        set(CLANG_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3} PARENT_SCOPE)
    endif()

endfunction()

# finds the most recent version of an LLVM executable installed on the system
function(_find_most_recent_llvm_executable EXECUTABLE_NAME)
    # capitalize and convert dashes to underscored for variable name
    string(TOUPPER ${EXECUTABLE_NAME} VARNAME)
    string(REPLACE "-" "_" VARNAME ${VARNAME})

    find_program(
        ${VARNAME}
        NAMES ${EXECUTABLE_NAME}-12.0
              ${EXECUTABLE_NAME}120
              ${EXECUTABLE_NAME}-12
              ${EXECUTABLE_NAME}-11.1
              ${EXECUTABLE_NAME}111
              ${EXECUTABLE_NAME}-11.0.1
              ${EXECUTABLE_NAME}1101
              ${EXECUTABLE_NAME}-11.0
              ${EXECUTABLE_NAME}110
              ${EXECUTABLE_NAME}-11
              ${EXECUTABLE_NAME}-10.0.1
              ${EXECUTABLE_NAME}1001
              ${EXECUTABLE_NAME}-10.0
              ${EXECUTABLE_NAME}100
              ${EXECUTABLE_NAME}-10
              ${EXECUTABLE_NAME}-9.0.1
              ${EXECUTABLE_NAME}901
              ${EXECUTABLE_NAME}-9.0
              ${EXECUTABLE_NAME}90
              ${EXECUTABLE_NAME}-9
              ${EXECUTABLE_NAME}-8.0.1
              ${EXECUTABLE_NAME}801
              ${EXECUTABLE_NAME}-8.0
              ${EXECUTABLE_NAME}80
              ${EXECUTABLE_NAME}-8
              ${EXECUTABLE_NAME}
        DOC "Path to ${EXECUTABLE_NAME}"
    )
endfunction()

# determine where libcxx is installed by parsing the output of the clang compilation stage
function(_determine_path_to_libcxx CLANG_EXECUTABLE)
    set(NOOP_CPP ${CMAKE_BINARY_DIR}/noop.cpp)
    if(NOT EXISTS ${NOOP_CPP})
        file(WRITE ${NOOP_CPP} "int main() {}")
    endif()

    execute_process(
        COMMAND ${CLANG_EXECUTABLE} -v -c -stdlib=libc++ ${NOOP_CPP}
        ERROR_VARIABLE CLANG_OUT
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    string(REPLACE "\n" ";" CLANG_OUT ${CLANG_OUT})

    set(LIBCXX_INCLUDE_DIR LIBCXX_INCLUDE_DIR-NOTFOUND CACHE INTERNAL "path to the libcxx STL")
    set(REACHED_INCLUDE_PATH FALSE)
    foreach(LINE ${CLANG_OUT})
        if(REACHED_INCLUDE_PATH)
            if(LINE STREQUAL "End of search list.")
                set(REACHED_INCLUDE_PATH FALSE)
            else()
                string(STRIP ${LINE} LINE)
                # LINE is an include path, check whether it is the C++ STL by looking for iostream
                if(EXISTS ${LINE}/iostream)
                    set(LIBCXX_INCLUDE_DIR ${LINE} CACHE INTERNAL "path to the libcxx STL")
                    break()
                endif()
            endif()
        endif()

        if(LINE STREQUAL "#include <...> search starts here:")
            set(REACHED_INCLUDE_PATH TRUE)
        endif()
    endforeach()
endfunction()

if(NOT LLVM_CONFIG)
    # llvm-config not manually set and we are using Clang as the compiler, select the same llvm version as clang,
    # instead of the most recent one
    if(CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        set(CLANG ${CMAKE_C_COMPILER} CACHE INTERNAL "Path to clang executable")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        set(CLANG ${CMAKE_CXX_COMPILER} CACHE INTERNAL "Path to clang executable")
    endif()

    if(CLANG)
        # determine clang version
        _set_clang_version(${CLANG})

        if(CLANG VERSION_GREATER_EQUAL "8")
            find_program(
                LLVM_CONFIG
                NAMES llvm-config-${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}
                      llvm-config-${CLANG_VERSION_MAJOR}${CLANG_VERSION_MINOR}${CLANG_VERSION_PATCH}
                      llvm-config-${CLANG_VERSION_MAJOR} llvm-config
            )
        endif()

        _determine_path_to_libcxx(${CLANG})
    endif()
endif()

# if LLVM_CONFIG not set manually and couldn't be deduced, look for most recent one
if(NOT LLVM_CONFIG)
    _find_most_recent_llvm_executable(llvm-config)
endif()

if(LLVM_CONFIG)
    execute_process(COMMAND ${LLVM_CONFIG} --version OUTPUT_VARIABLE LLVM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

    # try to find the same version clang as llvm-config
    if(NOT CLANG OR (NOT CLANG_VERSION VERSION_EQUAL LLVM_VERSION))
        string(REGEX MATCH "([0-9]+)\.([0-9]+)\.([0-9]+)" REGEX_MATCH ${LLVM_VERSION})
        set(LLVM_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(LLVM_VERSION_MINOR ${CMAKE_MATCH_2})
        set(LLVM_VERSION_PATCH ${CMAKE_MATCH_3})

        find_program(
            CLANG
            NAMES clang-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}
                  clang-${LLVM_VERSION_MAJOR}${LLVM_VERSION_MINOR}${LLVM_VERSION_PATCH} clang-${LLVM_VERSION_MAJOR}
                  clang
        )

        if(CLANG)
            _set_clang_version(${CLANG})
            _determine_path_to_libcxx(${CLANG})
        endif()
    endif()

    if(CLANG_VERSION VERSION_EQUAL LLVM_VERSION)
        # if we have found matching version of llvm and clang, use clang to determine the clang resource dir
        execute_process(
            COMMAND ${CLANG} -print-resource-dir OUTPUT_VARIABLE CLANG_RESOURCE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(CLANG_RESOURCE_DIR ${CLANG_RESOURCE_DIR} CACHE INTERNAL "path to Clang resource directory")
    else()
        # if the versions are not matching, we have to use the default location
        get_filename_component(LLVM_CONFIG_DIR ${LLVM_CONFIG} DIRECTORY)
        set(CLANG_RESOURCE_DIR "${LLVM_CONFIG_DIR}/../lib/clang/${LLVM_VERSION}"
            CACHE INTERNAL "path to Clang resource directory"
        )
        # issue a warning since this doesn't always work (see #78)
        message(WARNING "CLANG_RESOURCE_DIR set to default: ${CLANG_RESOURCE_DIR}, you may have to set it manually")
    endif()

    execute_process(COMMAND ${LLVM_CONFIG} --cxxflags OUTPUT_VARIABLE LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

    separate_arguments(LLVM_CXXFLAGS NATIVE_COMMAND ${LLVM_CXXFLAGS})

    execute_process(
        COMMAND ${LLVM_CONFIG} --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(COMMAND ${LLVM_CONFIG} --libdir OUTPUT_VARIABLE LLVM_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

    find_library(
        LLVM_LIBRARY
        NAMES LLVM
        PATHS ${LLVM_LIB_DIR}
        NO_DEFAULT_PATH
    )

    include(FindPackageHandleStandardArgs)
    set(IS_LLVM_STATICALLY_LINKED "YES")
    if(NOT LLVM_VERSION VERSION_LESS "9.0")
        find_library(
            CLANG_CPP_LIBRARY
            NAMES clang-cpp
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        if(CLANG_CPP_LIBRARY)
            # TODO: get_prerequisites is deprecated, GET_RUNTIME_DEPS throws dev-warning and fails linting
            include(GetPrerequisites)
            get_prerequisites(${CLANG_CPP_LIBRARY} LLVM_DEPS 0 0 "" "")
            # file(GET_RUNTIME_DEPENDENCIES  RESOLVED_DEPENDENCIES_VAR LLVM_DEPS EXECUTABLES ${LLVM_CONFIG})
            foreach(LLVM_DEP IN LISTS LLVM_DEPS)
                if(${LLVM_DEP} MATCHES "/libstdc[+][+][.]")
                    set(LLVM_CXX_STDLIB_DEP stdc++)
                elseif(${LLVM_DEP} MATCHES "/libc[+][+][.]")
                    set(LLVM_CXX_STDLIB_DEP c++)
                elseif(${LLVM_DEP} MATCHES "/libLLVM(-[0-9]+)?[.]")
                    set(IS_LLVM_STATICALLY_LINKED "NO")
                endif()
            endforeach()
        endif()

        # only require libLLVM if it is not statically linked into clang-cpp
        if(IS_LLVM_STATICALLY_LINKED)
            set(LLVM_LIB_VAR "")
        else()
            set(LLVM_LIB_VAR "LLVM_LIBRARY")
        endif()
        find_package_handle_standard_args(
            LLVM VERSION_VAR LLVM_VERSION REQUIRED_VARS LLVM_LIB_DIR LLVM_INCLUDE_DIR ${LLVM_LIB_VAR} CLANG_CPP_LIBRARY
                                                        CLANG_RESOURCE_DIR
        )
    else()
        # for LLVM versions older than 9 we expect all components to be static libaries
        find_library(
            CLANG_AST_LIBRARY
            NAMES clangAST
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_BASIC_LIBRARY
            NAMES clangBasic
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_FRONTEND_LIBRARY
            NAMES clangFrontend
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_TOOLING_LIBRARY
            NAMES clangTooling
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_DRIVER_LIBRARY
            NAMES clangDriver
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_LEX_LIBRARY
            NAMES clangLex
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_PARSE_LIBRARY
            NAMES clangParse
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_SEMA_LIBRARY
            NAMES clangSema
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_SERIALIZATION_LIBRARY
            NAMES clangSerialization
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_ANALYSIS_LIBRARY
            NAMES clangAnalysis
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(
            CLANG_EDIT_LIBRARY
            NAMES clangEdit
            PATHS ${LLVM_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_package_handle_standard_args(
            LLVM
            VERSION_VAR LLVM_VERSION
            REQUIRED_VARS
                LLVM_LIB_DIR
                LLVM_INCLUDE_DIR
                LLVM_LIBRARY
                CLANG_AST_LIBRARY
                CLANG_BASIC_LIBRARY
                CLANG_FRONTEND_LIBRARY
                CLANG_TOOLING_LIBRARY
                CLANG_DRIVER_LIBRARY
                CLANG_LEX_LIBRARY
                CLANG_PARSE_LIBRARY
                CLANG_SEMA_LIBRARY
                CLANG_SERIALIZATION_LIBRARY
                CLANG_ANALYSIS_LIBRARY
                CLANG_EDIT_LIBRARY
                CLANG_RESOURCE_DIR
        )
    endif()
else()
    message(WARNING "llvm-config could not be located")
endif()
