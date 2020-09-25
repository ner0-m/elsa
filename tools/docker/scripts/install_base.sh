#!/bin/bash
# Base packages
 
set -eux

# Install base images, such as git, compiler, git and python 
# Wrapped in (), to receive a single error code for the whole line if anything is wrong 
( apt-get update -qq && apt-get upgrade -qqy \
    && apt-get --no-install-recommends install -qq build-essential wget git ca-certificates libssl-dev \
       python3 python3-pip python3-setuptools python3-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && pip3 install wheel \
    && pip3 install numpy \
    && rm -rf /var/lib/apt/lists/* )
 
if [ "$?" != "0" ] ; then
    echo "Error installing base packages" 
    exit 1
fi
 
# Install ninja (yes ninja is just a binary, which is copied :-D) 
( apt-get update -qq \
    && git clone --single-branch --branch release https://github.com/ninja-build/ninja.git /tmp/ninja && cd /tmp/ninja \
    && ./configure.py --bootstrap \
    && cp ninja /usr/bin \
    && rm -rf /var/lib/apt/lists/* )
 
if [ "$?" != "0" ] ; then
    echo "Error building ninja"
    exit 1
fi
