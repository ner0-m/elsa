#!/bin/bash
# Base packages

( apt-get update -qq \
    && apt-get --no-install-recommends install -qq build-essential wget git ca-certificates ninja-build libssl-dev \
       python3 python3-pip python3-setuptools python3-dev \
    && pip3 install wheel \
    && pip3 install conan numpy \
    && rm -rf /var/lib/apt/lists/* )

if [ "$?" != "0" ] ; then
  exit 1
fi
