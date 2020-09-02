#!/bin/bash

exit_flag=false
 
target_branch="master"
 
# Retrieve list of cpp-files that were changed in source branch with respect to master (target branch)
filelist=($(git diff origin/${target_branch} --name-only | grep ".cpp"))
 

if [[ "${#filelist[@]}" -eq "0" ]]; then
    echo "==> No cpp files found"
    echo "==> clang-tidy has nothing to do, stop early"
    exit 0
else
    echo "==> Found ${#filelist[@]} cpp files"
    echo "==> Let's start our clang-tidy check"
fi

# for compilation database
mkdir -p build
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DELSA_CUDA_VECTOR=ON -DELSA_BENCHMARKS=ON
cd ..

echo
echo "clang-tidy checking changed files compared to target branch ${target_branch}"

# function to check if C++ file (based on suffix)
function checkCPP(){
    if [[ -f $1 ]] && [[ $1 == *.cpp ]]; then
        return 0
    fi
    return 1
}

clang-tidy-8 --version
echo


filesWithErrors=()

# check list of files
for f in $filelist; do
    # check if .cpp file and in compilation DB
    if checkCPP $f && [[ -n $(grep $f build/compile_commands.json) ]]; then
        echo "Checking matching file ${f}"
        touch output.txt
        clang-tidy-8 -p=build ${f} --extra-arg=--cuda-host-only > output.txt
        # decide if error or warning fail
        if [[ -n $(grep "warning: " output.txt) ]] || [[ -n $(grep "error: " output.txt) ]]; then
            echo ""
            echo "You must pass the clang tidy checks before submitting a pull request"
            echo ""
            grep --color -E '^|warning: |error: ' output.txt
            if [[ -n $(grep "error: " output.txt) ]]; then
                exit_flag=true
                filesWithErrors=( "${filesWithErrors[@]}" $f )
            fi
        else
            echo -e "\033[1;32m\xE2\x9C\x93 passed file $f\033[0m $1";
        fi
        rm output.txt
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
