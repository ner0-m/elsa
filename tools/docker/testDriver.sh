#!/bin/bash

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

OPTIONS=n:h:
LONGOPTS=network:,privileged,run-cuda,help

# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
 
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

usage()
{
    printf "Usage: %s [<options>] -- <image> <version> <script> [<pass to script>]\n" $(basename $0)
    echo ""
    echo "Available options:"
    echo "  -n | --network   mode   Network mode passed to docker command [host, bridge]"
    echo "       --privileged       Give extended privileged to this docker container"
    echo "       --run-cuda         Enable CUDA runtime for the docker container"
    echo "  -h | --help             Display this help"
}

network_mode=-
run_privileged=n
run_cuda=n 
 
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        --privileged)
            run_privileged=y
            shift
            ;;
        --run-cuda)
            run_cuda=y
            shift
            ;;
        -n|--network)
            network_mode=$2
            echo "Network mode: $network_mode"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0 
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "$1" 
            echo "Programming error"
            exit 3
            ;;
    esac
done

# handle non-option arguments (we need 3)
if [[ $# -lt 3 ]]; then
    echo "$0: 3 inputs (image name, image version and script) are necessary"
    exit 4
fi

IMAGE_NAME=$1
IMAGE_VERSION=$2
TEST_SCRIPT=$3
IMAGE=${IMAGE_NAME}:${IMAGE_VERSION}

# shift arguments
shift 3

DOCKER_ARGS=""

if docker network ls | grep -o "$network_mode" | sort | uniq &>/dev/null;then
    DOCKER_ARGS="${DOCKER_ARGS} --network=$network_mode"
fi

# For stuff like ASAN we need to run the container privileged 
if [ "$run_privileged" == "y" ]; then
    DOCKER_ARGS="${DOCKER_ARGS} --privileged"
fi

if [ "${run_cuda}" == "y" ]; then
    # Depending on Docker version, different tags are supportet, so test them and choose
    # For docker 18, we have to use "--runtime=nvidia" for docker 19 "--gpu 1" (for 1 GPU)
    if docker run ${DOCKER_ARGS} --gpus 1 -t staging/${IMAGE} echo "Test" &>/dev/null;then
        DOCKER_ARGS="${DOCKER_ARGS} --gpus 1"
    elif docker run ${DOCKER_ARGS} --runtime=nvidia -t staging/${IMAGE} echo "Test" &>/dev/null; then
        DOCKER_ARGS="${DOCKER_ARGS} --runtime=nvidia"
    else
        echo "This might be a problem, we can't run a docker with GPUs, please check!"
        exit 1
    fi
fi

echo "Testing ${IMAGE}..."
echo docker run ${DOCKER_ARGS} -t staging/${IMAGE} bash /tmp/tests/${TEST_SCRIPT} "$@"
docker run ${DOCKER_ARGS} -t staging/${IMAGE} bash /tmp/tests/${TEST_SCRIPT} "$@"
if [ "$?" != "0" ] ; then
        echo "Failed building image ${IMAGE}"
	exit 1
else
    echo "Successfuly build image ${IMAGE}"
    docker image tag staging/${IMAGE} elsa/${IMAGE}
    docker image tag elsa/${IMAGE} elsa/${IMAGE_NAME}:latest
    docker rmi staging/${IMAGE}
fi
