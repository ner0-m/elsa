# This script converts the commands in a compile_commands.json to a version that clang understands and saves the result
# in the directory specified by OUTPUT_DIR The variables CMAKE_BINARY_DIR and CLANG (path to clang executable) must be
# defined Conversion must happen after CMake generation step and prior to invoking the pybind11_generator

# parse all permissible clang flags from the output of "clang --help"
execute_process(COMMAND ${CLANG} --help OUTPUT_VARIABLE CLANG_OUT)
string(REGEX MATCHALL "\n[ \t]+--?[+-9A-Za-z_]+" CLANG_ALLOWED_FLAGS ${CLANG_OUT})
list(TRANSFORM CLANG_ALLOWED_FLAGS REPLACE "\n[ \t]+" "")
# flags comprised of only a single capital letter require special attention no " " or "=" between the flag and
# corresponding parameter, e.g. "-I/elsa/core" is perfectly valid
string(REGEX MATCHALL "\n[ \t]+-[A-Z] ?<[^>]+>" CAPITAL_LETTER_FLAGS ${CLANG_OUT})
list(TRANSFORM CAPITAL_LETTER_FLAGS REPLACE "\n[ \t]+-([A-Z]) ?<[^>]+>" "\\1")
string(REPLACE ";" "" CAPITAL_LETTER_FLAGS ${CAPITAL_LETTER_FLAGS})

# determine all flags used in the "compile_commands.json"
file(READ ${COMPILE_COMMANDS} ORIGINAL_COMMANDS)
string(REGEX MATCHALL " --?[+-9A-Za-z_]+" FLAGS_TO_REMOVE ${ORIGINAL_COMMANDS})
list(REMOVE_DUPLICATES FLAGS_TO_REMOVE)
list(TRANSFORM FLAGS_TO_REMOVE REPLACE " " "")

# filter out flags flags supported by clang
list(FILTER FLAGS_TO_REMOVE EXCLUDE REGEX "-[${CAPITAL_LETTER_FLAGS}].+")
list(REMOVE_ITEM FLAGS_TO_REMOVE ${CLANG_ALLOWED_FLAGS})

# remove all remaining flags and attached parameters from the compile commands
set(MODIFIED_COMMANDS ${ORIGINAL_COMMANDS})
foreach(FLAG IN LISTS FLAGS_TO_REMOVE)
    string(REGEX REPLACE " ${FLAG}(( [^-][^ \"]+)|(=[^ \"]+))?" "" MODIFIED_COMMANDS ${MODIFIED_COMMANDS})
endforeach()

# modify flags that behave differently between NVCC and clang
string(REPLACE " -x cu" " -x cuda" MODIFIED_COMMANDS ${MODIFIED_COMMANDS})
string(REPLACE " -isystem=" " -isystem " MODIFIED_COMMANDS ${MODIFIED_COMMANDS})
file(WRITE ${OUTPUT_DIR}/compile_commands.json ${MODIFIED_COMMANDS})
