#!/bin/bash
set +o noclobber
#set -o pipefail -e

# Input Arguments 
APP_NAME=$1
metric=$2
value=$3
ngpu=$4
xtraTags="$5"
output=$6
sudoOpt=$7
xtraFields="$8"


# Check if ROCm Env Var is set
if [ -z ${ROCM_PATH} ]; then
  echo "ROCM_PATH is unset. Please set to location of ROCm install";
  exit
else 
  echo "ROCM_PATH is set to '${ROCM_PATH}'";
fi

MY_PATH="$(dirname -- "${BASH_SOURCE[0]}")"

source cmds.sh

TOKEN=${TOKEN:-$(cat token.txt)}

# Collect System Information
SERVER_ADDRESS=http://10.194.116.44:8086/write?db=jenkins   # Address is hardcoded and passed in from Jenkinsfile template

TIMESTAMPNS=`date +%s%N`

# Saving result to bash cmd script
if [ -z "$xtraFields" ]; then
  echo "curl -i -XPOST "$SERVER_ADDRESS" --header \"Authorization: Token $TOKEN\" --data-binary  \"$APP_NAME,ROCm=$ROCm,OS=$OS,Kernel=$KERNEL,AMDGPU=$AMDGPU,Node=$NODE,Repo=$REPO,Branch=$BRANCH,Commit=$COMMIT,cpu=$CPU,gpu=$GPU,sbios=$SBIOS,vbios=$VBIOS,GFXCLK=$GFXCLK,UCLK=$UCLK,ngpu=$ngpu,$xtraTags $metric=$value $TIMESTAMPNS\"" >> curl-cmds.sh
else
  echo "curl -i -XPOST "$SERVER_ADDRESS" --header \"Authorization: Token $TOKEN\" --data-binary  \"$APP_NAME,ROCm=$ROCm,OS=$OS,Kernel=$KERNEL,AMDGPU=$AMDGPU,Node=$NODE,Repo=$REPO,Branch=$BRANCH,Commit=$COMMIT,cpu=$CPU,gpu=$GPU,sbios=$SBIOS,vbios=$VBIOS,GFXCLK=$GFXCLK,UCLK=$UCLK,ngpu=$ngpu,$xtraTags $metric=$value,$xtraFields\"" >> curl-cmds.sh
fi

# If requested, run the curl command directly and send the result to the database
if [ "$output" = "1" ]; then
  curl -i -XPOST "$SERVER_ADDRESS" --data-binary  "$APP_NAME,ROCm=$ROCm,OS=$OS,Kernel=$KERNEL,AMDGPU=$AMDGPU,Node=$NODE,Repo=$REPO,Branch=$BRANCH,Commit=$COMMIT,cpu=$CPU,gpu=$GPU,sbios=$SBIOS,vbios=$VBIOS,GFXCLK=$GFXCLK,UCLK=$UCLK,ngpu=$ngpu,$xtraTags $metric=$value,$xtraFields"
fi

echo "Data collected and stored in ${PWD}/curl-cmds.sh"
echo "Please make the curl-cmds.sh executable and then run the script to send the results to the database"
