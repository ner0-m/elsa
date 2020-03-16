#!/bin/bash

FILES="build/elsa/core/tests/test_ExpressionTemplates"

if type cuda-memcheck &> /dev/null; then
    echo "cuda-memcheck available in version:"
    cuda-memcheck --version
else
    echo "cuda-memcheck not correctly installed"
    exit 1
fi

for f in $FILES
do
    echo
    echo "Processing $f"
    if [ ! -x $f ]; then
        echo "File $f not available, error!"
        exit 1
    else
        cuda-memcheck --leak-check full --error-exitcode 1 $f
        if [ $? -ne 0 ]; then
            echo
            echo "cuda-memcheck errors reported"
            exit 1
        fi
    fi
done

echo
echo "All cuda-memcheck tests pass"
exit 0
