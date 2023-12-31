#!/bin/bash
# set exit on error
set -e
exit_flag=false

target_branch="master"

# Retrieve list of cpp-files that were changed in source branch with respect to master (target branch)
# Prevent script from aborting if nothing is found by adding "|| :" after grep
filelist=($(git diff origin/${target_branch} --name-only | grep -E "\.(h|hpp|cpp)" || :))

clang_tidy_version=14
clang_tidy_tool_candiates=("clang-tidy-${clang_tidy_version}" "clang-tidy")

if [[ "${#filelist[@]}" -eq "0" ]]; then
    echo "==> No .h, .hpp or .cpp files which are different to master found"
    echo "==> clang-tidy has nothing to do, stop early"
    exit 0
else
    echo "==> Found ${#filelist[@]} cpp files"
    echo "==> ${filelist[*]}"
    echo "==> Let's start our clang-tidy check"
fi

# Check all possible candidates
for candidate in ${clang_tidy_tool_candiates[@]}; do
    find_str=" ${clang_tidy_version}."
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" --version | grep -qF $find_str; then
        clang_tidy_tool=$candidate
        break
    fi
done

if [ ! -v clang_tidy_tool ];then
    echo "The desired \"clang-tidy-${clang_tidy_version}\" was not found. Aborting"
    exit 1
fi

# for compilation database
mkdir -p build
cd build
CC=clang CXX=clang++ cmake ..
cd ..

if [ ! -f build/compile_commands.json ]; then
    echo "Compilation database not found!"
    echo "--> Most likely the configuration failed"
    exit 1
fi

echo
echo "clang-tidy checking changed files compared to target branch ${target_branch}"

# function to check if C++ file (based on suffix)
function checkCPP(){
    if [[ -f $1 ]] && [[ $1 == *.cpp ]]; then
        return 0
    fi
    return 1
}

$clang_tidy_tool --version
echo


filesWithErrors=()

# check list of files
for f in ${filelist[*]}; do
    # check if .cpp file and in compilation DB
    if checkCPP $f && [[ -n $(grep $f build/compile_commands.json) ]]; then
        echo "Checking matching file ${f}"
        touch output.txt
        # Prevent script from aborting if "clang_tidy_tool" returns an non zero exit code by appending "|| :" in the end
        $clang_tidy_tool -p=build ${f} --extra-arg=--cuda-host-only > output.txt ||:

        # decide if error or warning fail
        if [[ -n $(grep "warning: " output.txt) ]] || [[ -n $(grep "error: " output.txt) ]]; then
            echo ""
            echo "You must pass the clang tidy checks before submitting a pull request"
            echo ""
            grep --color -E '^|warning: |error: ' output.txt
            exit_flag=true
            filesWithErrors=( "${filesWithErrors[@]}" $f )
            if [[ -n $(grep "error: " output.txt) ]]; then
                echo -e "\033[1;31m\xE2\x9C\x98 failed file $f\033[0m $1";
            else
                echo -e "\033[1;31m\xE2\x9C\x98 failed file $f because of warnings\033[0m $1";
            fi
        else
            echo -e "\033[1;32m\xE2\x9C\x93 passed file $f\033[0m $1";
        fi
        rm output.txt
    else
        echo "$f not a C++ file or not in compilation database (compile_commands.json)"
    fi
done


if [ "$exit_flag" = true ]; then
    echo "Error in file(s):"
    for f in "${filesWithErrors[@]}"; do
        echo "$f"
    done
    exit -1
fi

echo "clang-tidy check passed"

exit 0
