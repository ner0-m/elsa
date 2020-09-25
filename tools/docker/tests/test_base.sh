#!/bin/bash

set -e

# Just call some of the tools that should be installed 
echo "=== git ===" 
git --version 
echo ""
 
echo "=== GCC ===" 
gcc --version 
echo ""

echo "=== python ===" 
python --version
echo ""
 
echo "=== pip ===" 
pip --version 
echo ""
 
echo "=== ninja ===" 
ninja --version
echo ""
