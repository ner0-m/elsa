#!/bin/bash

# Format all CMake-like files with cmake-format
# Determine the applied differences with git,
# return 1 when changes had to be made, so the CI step fails.

RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

files=($(git diff origin/master --name-only --diff-filter=d | egrep ".*(CMakeLists.txt|\.cmake(\.in)?)$"))

if (( ${#files[@]} )); then
    cmake_lint_tool=false
    if command -v "cmake-format" >/dev/null 2>&1; then
        cmake_lint_tool=cmake-format
    else
        echo -e "[${RED}FAIL${NC}]: cmake-format is not available, but CMake files need linting! Please install cmake-format"
        exit 1
    fi

    echo -e "[${BLUE}INFO${NC}]: Formatting check with: `$cmake_lint_tool --version`"
    echo -e "[${BLUE}INFO${NC}]: Running '$cmake_lint_tool --check' on files..."

    if ! "$cmake_lint_tool" --check "${files[@]}"; then
        echo -e "[${RED}FAIL${NC}]: Ups, something isn't correct with the formatting, please check above errors"
        echo -e "[${BLUE}INFO${NC}]: From the root directory you can also run:"
        echo "find CMakeLists.txt elsa benchmarks examples tools cmake -name '*.cmake' -o -name 'CMakeLists.txt' | xargs $cmake_lint_tool -i"
        exit 1
    else
        echo -e "[${GREEN}OK${NC}]: Excellent. Formatting check passed"
    fi
else
    echo -e "[${GREEN}OK${NC}]: No CMake files to check"
fi
