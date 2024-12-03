#!/bin/bash
#
# Start an interactive process within a docker container for a project by:
#   * Building a docker image (if needed)
#   * Running the docker image in a container
#
# With:
#   * Automatic image/container naming:
#     * Image: ${project_name}.dev
#     * Container: ${Image}-${USER}-${TIMESTAMP}
#   * Automatic volume mounts:
#     * $project_dir:/workdir
#     * $datadir:/data
#   * Automatic environment variables:
#     * DATADIR=/data
#     * PROJECT=$project_name
#   * Automatic linked docker network
#
# Args:
#
# --entrypoint : The entrypoint within the container [default="/bin/bash"]
# --project_dir : The project directory [default $PWD]
# --dockerfile : The dockerfile to build and run [default "${project_dir}/docker/Dockerfile.dev|prod"]
# --project_name : The project name (identifies main project package directory and used for tagging the docker container) [default $(basename $project_dir) with dashes changed to underscores]
# -p : Port mapping to add (host:container)
# -v : Volume mapping to add (host:container)
# -e : Environment mapping to add (var:value)
# --gpus : The GPU(s) to use IF available (default "all")
# --network_name : The network name [default "devnet"]
# --user : The user (for tagging the docker container) [default $USER]
# --datadir : The (outside of the container) data directory [default $DATADIR or $HOME/data]
# --time_format: The timestamp format to use for the container name [default "+%Y-%m-%d_%H.%M.%S"]
# --timestamp: A timestamp to use, overriding "now"
# --rebuild:  Option present to force a rebuild of the image
# --prod: Option to build/run the production image instead of development
# --quiet: Option present to suppress verbosity
# --daemon: Run container non-interactively
# --dryrun:  Option present for dry run (no build or run)
#

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

entrypoint="/bin/bash"
project_dir="${PWD}"
dockerfile=""
project_name=""
name_override=""
gpus="all"
network_name=devnet
user="$USER"
datadir="${DATADIR:=$HOME/data}"
time_format="+%Y-%m-%d_%H.%M.%S"
timestamp=""
rebuild=""
build_args=""
env="dev"
verbose="y"
daemon=""
dryrun=""

DOCKER_ARGS=""

DOCKER_CMD="docker"
if test -n "$(uname -a | grep -i linux)"; then
    DOCKER_CMD="sudo docker"
fi

# parse args
while [[ $# > 0 ]]
do
    key="$1";

    case "${key}" in
        --entrypoint)
            entrypoint="$2";
            shift;
            ;;
        --project_dir)
            project_dir="$2";
            shift;
            ;;
        --dockerfile)
            dockerfile="$2";
            shift;
            ;;
        --project_name)
            project_name="$2";
            shift;
            ;;
        --name_override)
            name_override="$2";
            shift;
            ;;
        -[pve])
            DOCKER_ARGS="${DOCKER_ARGS} $1 $2";
            shift;
            ;;
        --gpus)
            gpus="$2";
            shift;
            ;;
        --network_name)
            network_name="$2";
            shift;
            ;;
        --user)
            user="$2";
            shift;
            ;;
        --datadir)
            datadir="$2";
            shift;
            ;;
        --time_format)
            time_format="$2";
            shift;
            ;;
        --timestamp)
            timestamp="$2";
            shift;
            ;;
        --rebuild)
            rebuild="y";
            ;;
        --build-arg)
            build_args="${build_args} $1 $2";
            shift;
            ;;
        --prod)
            env="prod";
            ;;
        --quiet)
            verbose="";
            ;;
        --daemon)
            daemon="y";
            ;;
        --dryrun)
            dryrun="y";
            verbose="y";
            ;;
        *)
            DOCKER_ARGS="${DOCKER_ARGS} $1";
            ;;
    esac;
    shift;
done


if test -z "${daemon}"; then
    DOCKER_ARGS="--init --rm -it ${DOCKER_ARGS}"
fi

# Add network
test -n "${network_name}" && DOCKER_ARGS="${DOCKER_ARGS} --net ${network_name}"

# Set dockerfile
test -z "${dockerfile}" && dockerfile="${project_dir}/docker/Dockerfile.${env}"
test -n "${verbose}" && echo "dockerfile=$dockerfile"

# Set project name
test -z "${project_name}" && project_name="$(basename ${project_dir} | sed s/-/_/g)"
test -n "${verbose}" && echo "project_name=$project_name"

# Drop "gpus" if there isn't a GPU
if test -n "${gpus}"; then
   NVSMI="$(nvidia-smi > /dev/null; echo $?)"
   if [ ${NVSMI} -ne 0 ]; then
       echo "nvidia-smi error. Dropping gpus ($gpus)"
       gpus=""
   else
       echo "adding gpus ($gpus) to docker args"
       DOCKER_ARGS="${DOCKER_ARGS} --gpus ${gpus}"
   fi
fi
test -n "${verbose}" && echo "gpus=$gpus (NVSMI=$NVSMI)"

test -n "${verbose}" && echo "datadir=$datadir"

# Add automatic volumes and environment variables
if test -n "${project_dir}"; then
    if [ ${env} != 'prod' ]; then
        DOCKER_ARGS="${DOCKER_ARGS}  \
        -v ${project_dir}:/workdir"
    fi
    DOCKER_ARGS="${DOCKER_ARGS}  \
        -e DATADIR=/data"
fi
   
if test -n "${datadir}"; then
    DOCKER_ARGS="${DOCKER_ARGS}  \
        -v ${datadir}:/data"
    DOCKER_ARGS="${DOCKER_ARGS}  \
        -e PROJECT=${project_name}"
fi


# Set timestamp
test -z "${timestamp}" && timestamp=$(date "${time_format}")
test -n "${verbose}" && echo "timestamp=$timestamp"

# Build docker image if needed
if test -n ${rebuild} || test -z "$($DOCKER_CMD images | grep ${project_name})"; then
    echo "Building ${project_name}..."
    if test -n "${verbose}"; then
        echo "$DOCKER_CMD build --compress \
            -t ${project_name}.${env} \
            -f ${dockerfile} .";
    fi

    if test -z "${dryrun}"; then
        $DOCKER_CMD build --compress \
            -t ${project_name}.${env} \
            -f "${dockerfile}" .;
    fi
    echo "...done building ${project_name}"
fi

# Create docker network if needed
test -z "$($DOCKER_CMD network list | grep ${network_name})" && $DOCKER_CMD network create --attachable ${network_name}
test -n "${verbose}" && echo "network_name=$network_name"

# Add name arg
test -z ${name_override} && name_override=${project_name}
DOCKER_ARGS="${DOCKER_ARGS}  \
     --name ${name_override}.${env}-${user}-${timestamp}"

# Add entrypoint arg
if test -n "${entrypoint}"; then
    DOCKER_ARGS="${DOCKER_ARGS}  \
         --entrypoint ${entrypoint}"
fi

# Add image name to args
DOCKER_ARGS="${DOCKER_ARGS}  \
     ${project_name}.${env}:latest"

test -n "${verbose}" && echo "DOCKER_ARGS=$DOCKER_ARGS"

# Run docker container
if test -n "${verbose}"; then
    echo "$DOCKER_CMD run ${DOCKER_ARGS}";
fi
if test -z "${dryrun}"; then
   $DOCKER_CMD run ${DOCKER_ARGS}
fi
