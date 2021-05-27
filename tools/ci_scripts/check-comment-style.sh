#!/bin/bash

# Retrieve list of cpp-files that were changed in source branch with respect to master (target branch)
target_branch="master"
filelist=($(git diff origin/${target_branch} --name-only | egrep ".+\.(h|cpp|hpp|cu|cuh)"))

if [[ "${#filelist[@]}" -eq "0" ]]; then
    echo "--> No relevant files changed compared to $target_branch branch, all good!"
    exit 0
else
    echo "--> Found ${#filelist[@]} relevant files, which changed compared to $target_branch branch"
fi

# This list is obviously incomplete but should be fine for now and at least help us a little bit!
# This should have very few false negatives and I prefer that.
grepoutput=$(grep --color=always -re '\s\* \\\(brief\|author\|param\|tparam\|warning\|return\|returns\|file\|throw\)\(\[[a-z]\+\]\)\?\s\+' elsa/)

if [[ $(printf "%s" "$grepoutput" | wc -l) -gt 0 ]]; then
    echo -e "\e[33mWarning:\e[0m Found files that use backslash instead of at for doxygen tags"
    echo "--> Replace for a single file: \`sed -i -e 's/\\\\<tok>/@<tok>/g' file\`"
    echo "--> Replace all with 'find elsa -type f \( -iname \*.h -o -iname \*.cpp -o -iname \*.hpp -o -iname \*.cuh -o -iname \*.cu \) \\
    -exec sed -i -e 's/\\\\brief/@brief/g' -e 's/\\\\author/@author/g' \\
    -e 's/\\\\param/@param/g' -e 's/\\\\tparam/@tparam/g' -e 's/\\\\warning/@warning/g' \\
    -e 's/\\\\return/@return/g'  -e 's/\\\\returns/@returns/g' -e 's/\\\\file/@file/g' \\
    -e 's/\\\\throw/@throw/g' {} \\;'"
    echo "--> or 'fd -e h -e cpp -e cu -e cuh -x sed -i -e 's/\\\\brief/@brief/g' \\
    -e 's/\\\\author/@author/g' -e 's/\\\\param/@param/g' -e 's/\\\\tparam/@tparam/g' \\
    -e 's/\\\\warning/@warning/g'  -e 's/\\\\return/@return/g'  -e 's/\\\\returns/@returns/g' \\
    -e 's/\\\\file/@file/g' -e 's/\\\\throw/@throw/g''"
    echo -e "--> \e[33mBe aware that this can also be a false negative\e[0m"
    echo ""
    echo "Problematic places:"
    echo "$grepoutput"
    echo ""
    echo "Please check the above, if there are any doxygen tags you can replace"
    exit 1 
else
    echo -e "\e[32mEverything is perfect!\e[0m"
fi
