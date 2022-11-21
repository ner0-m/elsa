#!/bin/bash

# Check for all C++-like files if they need formatting.
# Only consider files which are different from master.
# clang-format version 10 is required.
# return 1 when changes have to be made, so the CI step fails.

RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color

# preference list
clang_format_version=14
clang_format_tool_candiates=(clang-format-$clang_format_version clang-format)

# update origin
git fetch origin
master_ahead=$(git log HEAD..origin/master --oneline | wc -l)

if [[ "$master_ahead" -gt "0" ]]; then
    echo -e "[${ORANGE}WARNING${NC}]: Not all commits which are part of origin/master are part of your branch, maybe you need to rebase?"
fi

files=($(git diff origin/master --name-only --diff-filter=d | egrep ".+\.(h|hpp|cpp|cu|cuh)$"))

if (( ${#files[@]} )); then
    clang_format_tool=false
    # Check all possible candidates
    for candidate in ${clang_format_tool_candiates[@]}; do
        find_str=" ${clang_format_version}."
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" --version | grep -qF $find_str; then
            clang_format_tool=$candidate
            break
        fi
    done

    # Check that it's really set
    if [ $clang_format_tool = false ]; then
        echo -e "[${RED}FAIL${NC}]: clang-format-$clang_format_version is not available, but C++ files need linting! Please install clang-format-$clang_format_version."
        exit 1
    fi

    echo -e "[${BLUE}INFO${NC}]: Formatting check with: `$clang_format_tool --version`"
    echo -e "[${BLUE}INFO${NC}]: Running '$clang_format_tool --dry-run -Werror -style=file' on files..."

    # Dry run clang-format with -Werror set
    if ! "$clang_format_tool" --dry-run -Werror -style=file "${files[@]}"; then
        echo -e "[${RED}FAIL${NC}]: Oops, something isn't correct with the formatting, please check the errors above"
        echo -e "[${BLUE}INFO${NC}]: From the root directory you can also run:"
        echo "find elsa/ benchmarks/ examples/ | egrep \".+\.(h|hpp|cpp|cu|cuh)$\" | xargs $clang_format_tool -i -style=file"

        if [[ "$master_ahead" -gt "0" ]]; then
            echo -e "[${BLUE}INFO${NC}]: Files considered:"
            printf '--> %s\n' "${files[@]}"
        fi

        exit 1
    else
        echo -e "[${GREEN}OK${NC}]: Excellent. Formatting check passed"
    fi
else
    echo -e "[${GREEN}OK${NC}]: No C++-like files to check"
fi
