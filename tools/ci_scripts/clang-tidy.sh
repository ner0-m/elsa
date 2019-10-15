#!/bin/bash

exit_flag=false

# for compilation database
mkdir -p build
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cd ..

target_branch="master"
echo
echo "clang-tidy checking changed files compared to target branch ${target_branch}"

# function to check if C++ file (based on suffix)
function checkCPP(){
    if [[ $1 == *.cpp ]];then
        return 0
    fi
    return 1
}

apt-get update
apt-get install clang-tidy-8 -y

clang-tidy-8 --version
echo

# Retrieve list of files that were changed in source branch with respect to master (target branch)
filelist=`git diff origin/${target_branch} --name-only`

# check list of files
for f in $filelist; do
    if checkCPP $f; then
        echo "Checking matching file ${f}"
        # apply the clang-format script
        touch output.txt
        clang-tidy-8 -p=build ${f} > output.txt
        # decide if error or warning fail
        if [[ -n $(grep "warning: " output.txt) ]] || [[ -n $(grep "error: " output.txt) ]]; then
            echo ""
            echo "You must pass the clang tidy checks before submitting a pull request"
            echo ""
            grep --color -E '^|warning: |error: ' output.txt
            if [[ -n $(grep "error: " output.txt) ]]; then
                exit_flag=true
            fi
        else
            echo -e "\033[1;32m\xE2\x9C\x93 passed:\033[0m $1";
        fi
        rm output.txt
    fi
done

if [ "$exit_flag" = true ]; then
    exit -1
fi

echo "clang-tidy check passed"

exit 0
