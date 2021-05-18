#!/bin/bash

# Format all CMake-like files with cmake-format
# Determine the applied differences with git,
# return 1 when changes had to be made, so the CI step fails.

# set exit on error
set -e

# preference list
cmake_format_tool_candiates=(cmake-format)

for candidate in ${cmake_format_tool_candiates[@]}; do
    if command -v "$candidate" >/dev/null; then
        echo "Formatting check with $candidate --version:"
        $candidate --version
        cmake_format_tool=$candidate
        break
    fi
done

if [[ -z "$cmake_format_tool" ]]; then
    echo "$cmake_format_tool not correctly installed"
    exit 1
fi

echo

# run cmake-format
format_call="find elsa benchmarks examples tools cmake -name '*.cmake' -o -name 'CMakeLists.txt' | xargs $cmake_format_tool -l error"

exit_code=0
eval $format_call --check || exit_code=$?

# if exit code 0, all is good
if [[ -z $exit_code ]]; then
    echo "Excellent. You passed the formatting check!"
    exit 0;
else
    echo
    echo "The above files have cmake-format problems"
    echo "Inside the repo, please run"
    echo
    echo "$format_call -i"
    echo
    echo "to solve the issue."
fi

exit 1
