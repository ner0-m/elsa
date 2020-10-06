#!/bin/bash

# Applies clang-format

# set exit on error 
set -e 
 
# check that we are in a clean state in order to prevent accidential changes
git status --untracked-files=no
if [ ! -z "$(git status --untracked-files=no  --porcelain)" ]; then
  echo "Script must be applied on a clean git state"
  exit 1
fi

echo

if type clang-format-10 2>/dev/null; then
    echo "Checking formatting using the following clang-format version:"
    clang-format-10 --version
else
    echo "clang-format-10 not correctly installed"
    exit 1
fi

echo

# perform clang-format on all cpp-files
find elsa/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.cu' -or -name '*.cuh' | xargs clang-format-10 -i -style=file $1
find benchmarks/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.cu' -or -name '*.cuh' | xargs clang-format-10 -i -style=file $1
find examples/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.cu' -or -name '*.cuh' | xargs clang-format-10 -i -style=file $1

# check if something was modified
notcorrectlist=`git status --porcelain | grep '^ M' | cut -c4-`
# if nothing changed ok
if [[ -z $notcorrectlist ]]; then
  # send a negative message to gitlab
  echo "Excellent. Very good formatting!"
  exit 0;
else
  echo "The following files have clang-format problems:"
  git diff --stat $notcorrectlist
  echo "Please run"
  echo
  echo "find elsa/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.cu' -or -name '*.cuh' | xargs clang-format-10 -i -style=file $1"
  echo "find benchmarks/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.cu' -or -name '*.cuh' | xargs clang-format-10 -i -style=file $1"
  echo "find examples/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.cu' -or -name '*.cuh' | xargs clang-format-10 -i -style=file $1"
  echo
  echo "to solve the issue."
  # cleanup changes in git
  git reset HEAD --hard
fi

exit 1
