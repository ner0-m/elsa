#!/bin/bash

set -ux
 
echo "=== clang-format ===" 

# Check that the version is actually correct, but this relies on grep, so maybe it's not perfect
clang-tidy-$1 -version | grep -c $1
 
if [ "$?" != "0" ] ; then
    echo "Error clang tidy version mismatch, expected $1" 
    exit 1
fi
echo "==="
