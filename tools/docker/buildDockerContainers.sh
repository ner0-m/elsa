#!/bin/sh

echo "Building docker image for Clang 6..."
docker build -t elsa/clang:6 - < DocFileClang6
echo "... building Clang 6 done."

echo "Building docker image for Clang 7..."
docker build -t elsa/clang:7 - < DocFileClang7
echo "... building Clang 7 done."

echo "Building docker image for Clang 8..."
docker build -t elsa/clang:8 - < DocFileClang8
echo "... building Clang 8 done."

echo "Building docker image for gcc 6..."
docker build -t elsa/gcc:6 - < DocFileGcc6
echo "... building gcc 6 done."

echo "Building docker image for gcc 7..."
docker build -t elsa/gcc:7 - < DocFileGcc7
echo "... building gcc 7 done."

echo "Building docker image for gcc 8..."
docker build -t elsa/gcc:8 - < DocFileGcc8
echo "... building gcc 8 done."

echo "Building docker image for gcc 9..."
docker build -t elsa/gcc:9 - < DocFileGcc9
echo "... building gcc 9 done."

echo "Building docker image for Cuda 9.2..."
docker build -t elsa/cuda:9.2 - < DocFileCuda92
echo "... building Cuda 9.2 done."

echo "Building docker image for Ubuntu 18.04..."
docker build -t elsa/ubuntu:18.04 - < DocFileUbuntu1804
echo "... building Ubuntu 18.04 done."

echo "Building docker image for Clang Tidy 8 (with CUDA 9.2)..."
docker build -t elsa/clang-tidy:8 - < DocFileClangTidy8
echo "... building Clang Tidy 8 (with CUDA 9.2) done."

