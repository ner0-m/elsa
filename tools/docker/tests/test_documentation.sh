#!/bin/bash

set -e

echo "Testing image ${1}:${2}..."

image=${1}:${2}

docker run -t staging/${image} /tmp/tests/test_elsa_doc.sh

if [ "$?" != "0" ] ; then
        echo "CMake based image has some error, please check"
        exit 1
else
        echo "... test passed"
        docker image tag staging/${image} elsa/${image}
        docker image tag staging/${image} elsa/${1}:latest
        docker rmi staging/${1}
fi

