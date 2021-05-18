#!/bin/bash

# Format all CMake-like files with cmake-format
# Determine the applied differences with git,
# return 1 when changes had to be made, so the CI step fails.

# set exit on error
set -e

# preference list
cmake_lint_tool_candiates=(cmake-lint)

for candidate in ${cmake_lint_tool_candiates[@]}; do
    if command -v "$candidate" >/dev/null; then
        echo "Linting check with $candidate --version:"
        $candidate --version
        cmake_lint_tool=$candidate
        break
    fi
done

if [[ -z "$cmake_lint_tool" ]]; then
    echo "$cmake_lint_tool not correctly installed"
    exit 1
fi

echo

# run cmake-lint
lint_call="find elsa benchmarks examples tools cmake -name '*.cmake' -o -name 'CMakeLists.txt' | xargs $cmake_lint_tool -l error"

exit_code=0
eval $lint_call || exit_code=$?

# if exit code 0, all is good
if [[ $exit_code -eq 0 ]]; then
    echo "Excellent. You passed the linting check!"
    exit 0;
else
    echo
    echo "The above files have cmake-lint problems"
    echo "Inside the repo, please run"
    echo
    echo "$lint_call"
    echo
    echo "to see the issue. Please check the output"
fi

exit 1
