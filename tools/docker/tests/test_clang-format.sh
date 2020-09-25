#!/bin/bash

set -ux
 
if [ -z "${1:-}" ]; then  
    echo "Need first argument, as version number for clang-tidy"
    exit 1
fi 
 
# We need git in our script so check for it 
echo "=== git ===" 
git --version 
echo "==="
 
echo "=== clang-format ==="
clang-format-$1 -version | grep -c $1
if [ "$?" != "0" ] ; then
    echo "Error clang tidy version mismatch, expected $1" 
    exit 1
fi
echo "==="
