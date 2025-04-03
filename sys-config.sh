#! /usr/bin/env bash

set +o noclobber
set -o pipefail

nosudo="0"
outfile=""
AGT="0"

GETOPT_PARSE=$(getopt --name "${0}" --longoptions nosudo,rocm:,file:,appname: --options n,r:,f:,a: -- "$@")
if [[ $? -ne 0 ]]; then
echo "getopt invocation failed; could not parse the command line";
exit 1
fi

eval set -- "${GETOPT_PARSE}"

while (( "$#" )); do
    case "$1" in
      -n|nosudo)
        nosudo="1"
        shift
        ;;
      -r|rocm)
        ROCM_PATH="$2"
        shift 2
        ;;
      -f|file)
        outfile="$2"
        shift 2
        ;;
      -a|appname)
        REPO_NAME="$2"
        shift 2
        ;;
      --) # end argument parsing
        shift
        break
        ;;
      -*|--*=|*) # unsupported flags
        echo "Error: Unsupported flag $1" >&2
        help
        exit 1
        ;;
    esac
done

# Check if ROCm Env Var is set
if [ -z ${ROCM_PATH} ]; then
  echo "ROCM_PATH is unset. Please set to location of ROCm install";
  exit
else 
  echo "ROCM_PATH is set to '${ROCM_PATH}'";
fi

if [[ -n $outfile ]]; then
    rm -f $outfile
else
    rm -f cmds.sh
fi

function set_var() {
    varname=$1
    value=$2
    command="$varname=$value"
    echo $command
    if [[ -z $outfile ]]; then
        echo "$command">>cmds.sh
    else
        echo "$command">>$outfile
    fi
}

# Collect System Information
set -o pipefail
OS=$(lsb_release -d | cut -c14- | tr -d '[:space:]' | tr -d "()")
if [[ $? -ne "0" ]]; then
  OS=$(cat /etc/os-release | grep PRETTY | cut -d\" -f 2 | tr -d '[:space:]' | tr -d "()")
fi
set +o pipefail
set_var OS "$OS"
set_var GFXCLK "n/a"
set_var UCLK "n/a"
set_var SBIOS "n/a"
set_var KERNEL "$(uname -r)"
set_var ROCm "$(cat $ROCM_PATH/.info/version)"             # For performance tracking (publish)
temp=($(dkms status | sed -e $'s/amdgpu\//\\\n/g' | sed -e $'s/,/\\\n/g'))
set_var AMDGPU "${temp[0]}"

if [ -d .git ]; then
   set_var REPO "$(basename `git rev-parse --show-toplevel`)"
   set_var BRANCH "$(git rev-parse --abbrev-ref HEAD)"
   set_var COMMIT "$(git rev-parse --short HEAD)"           # For performance tracking (publish)
else
   if [ -z ${REPO_NAME} ]; then
      echo "REPO_NAME is unset. Please set -reponame";
      exit
   fi
   set_var REPO ${REPO_NAME}
   set_var BRANCH "n/a"
   set_var COMMIT "n/a"
fi

set_var NODE "$(hostname)"             # For performance tracking (publish)
set_var CPU "$(cat /proc/cpuinfo | grep -m 1 'model name' | cut -c14- | tr -d '[:space:]' | tr -d '@()')"
temp=($(rocminfo | grep -m 1 gfx))
set_var GPU "${temp[1]}"
set_var VBIOS "$(rocm-smi -v | grep -F GPU[0] | grep -o version.* | cut -f2- -d: | xargs)"
