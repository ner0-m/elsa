#!/bin/bash
 
RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
 
# Retrieve list of cpp-files that were changed in source branch with respect to master (target branch)
target_branch="master"
filelist=($(git diff origin/${target_branch} --name-only | egrep ".+\.(h|hpp|cuh)"))

if [[ "${#filelist[@]}" -eq "0" ]]; then
    echo -e "[${GREEN}OK${NC}]: No C++-like headers to check" 
    exit 0
fi

# This list is obviously incomplete but should be fine for now and at least help us a little bit!
# This should have very few false negatives and I prefer that.
grepoutput=$(grep --color=always -re '\s\* \\\(brief\|author\|param\|tparam\|warning\|return\|returns\|file\|throw\)\(\[[a-z]\+\]\)\?\s\+' elsa/)

if [[ $grepoutput ]] || [[ $(printf "%s" "$grepoutput" | wc -l) -gt 0 ]]; then
    echo -e "[${RED}FAIL${NC}]: Found files that use backslash instead of at for doxygen tags"
    echo "$grepoutput"
    echo ""
    echo -e "[${BLUE}INFO${NC}]: Replace for a single file: \`sed -i -e 's/\\\\<tok>/@<tok>/g' file\`"
    echo -e "[${BLUE}INFO${NC}]: Replace all with 'find elsa -type f \( -iname \*.h -o -iname \*.cpp -o -iname \*.hpp -o -iname \*.cuh -o -iname \*.cu \) \\
    -exec sed -i -e 's/\\\\brief/@brief/g' -e 's/\\\\author/@author/g' \\
    -e 's/\\\\param/@param/g' -e 's/\\\\tparam/@tparam/g' -e 's/\\\\warning/@warning/g' \\
    -e 's/\\\\return/@return/g'  -e 's/\\\\returns/@returns/g' -e 's/\\\\file/@file/g' \\
    -e 's/\\\\throw/@throw/g' {} \\;'"
    echo -e "[${BLUE}INFO${NC}]: or 'fd -e h -e cpp -e cu -e cuh -x sed -i -e 's/\\\\brief/@brief/g' \\
    -e 's/\\\\author/@author/g' -e 's/\\\\param/@param/g' -e 's/\\\\tparam/@tparam/g' \\
    -e 's/\\\\warning/@warning/g'  -e 's/\\\\return/@return/g'  -e 's/\\\\returns/@returns/g' \\
    -e 's/\\\\file/@file/g' -e 's/\\\\throw/@throw/g''"
    echo -e "[${ORANGE}WARNING${NC}]: Be aware that this can also be a false negative"
    exit 1 
else
    echo -e "[${GREEN}OK${NC}]: Excellent. All tags use the correct styling" 
fi
