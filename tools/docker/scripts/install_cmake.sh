#!/bin/bash

set -e -u

CMAKE_VERSION=$1
CMAKE_DIR=$2

# Download and unpack
( wget https://cmake.org/files/v${CMAKE_VERSION}/${CMAKE_DIR}.tar.gz \
  && tar -xzf ${CMAKE_DIR}.tar.gz && rm ${CMAKE_DIR}.tar.gz && cd ${CMAKE_DIR} \
  && ./configure --parallel=28 --prefix=/tmp/cmake-install \
  && make -j28 && make -j28 install && cd .. && rm -rf ${CMAKE_DIR} && ldconfig )

if [ "$?" != "0" ] ; then
  exit 1
fi


