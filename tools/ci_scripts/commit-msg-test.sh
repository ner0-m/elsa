#!/usr/bin/env bash

# Small script to check that the commit message follows some clean formatting
# The following checks are in place:
# * Line length max 72 characters
# * Empty line after commit title
# * Commit title doesn't end with a period 
# The script expects the Commit message file as the first (and only) argument.
# This script is adapted from https://github.com/SerenityOS/serenity/blob/master/Meta/lint-commit.sh
# by the GitHub user IdanHo as part of the SerenityOS project.

RED='\033[0;31m'
NC='\033[0m' # No Color
 
if [ -z "$1" ]; then 
    echo -e "[${RED}FAIL${NC}]: Pass commit message file as first argument";
    exit 1 
fi 
 
# the file containing the commit message is passed as the first argument
commit_file="$1"
commit_message=$(cat "$commit_file")

error() {
    echo -e "[${RED}FAIL${NC}]: $1";
    echo "$commit_message"
    exit 1
}

# fail if the commit message contains windows style line breaks (carriage returns)
if grep -q -U $'\x0D' "$commit_file"; then
    error "Commit message contains CRLF line breaks (only unix-style LF linebreaks are allowed)"
fi

commentchar=$(git config --get core.commentChar)
re="^$commentchar -* >[0-9] -*\$" 
 
line_number=0
while read -r line; do
    # Somehow I need to make sure to not run till the diff... 
    [[ "$line" =~ $re ]] && break
    # ignore comment lines
    [[ "$line" =~ ^$commentchar.* ]] && continue
   
    ((line_number += 1))
    line_length=${#line}

    if [[ $line_number -eq 1 ]] && [[ "$line" =~ \.$ ]]; then
        error "Commit title ends in a period"
    fi
     
    if [[ $line_number -eq 2 ]] && [[ "$line" ]]; then
        error "The line after the commit message header should be empty"
    fi

    if [[ $line_length -gt 72 ]]; then
        error "Commit message lines are too long (maximum allowed is 72 characters)"
    fi
done <"$commit_file"
