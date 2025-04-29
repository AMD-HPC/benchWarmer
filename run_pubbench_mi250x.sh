#!/bin/bash
##SBATCH -w x1000c3s5b1n0
##SBATCH -w x1000c2s5b1n0
#SBATCH -w x1000c3s1b1n0
#SBATCH --gpus-per-node=1
#SBATCH --time=36:00:00
#SBATCH --output=./slurm_output/mi250x.out
#SBATCH --error=./slurm_output/mi250x.err

module load rocm

nOps=()
for ((i=0; i<16; i++)); do
    nOp=$(echo "2^$i" | bc)
    nOps+=($nOp)
done

ROCSTAR_ROOT="../rocSTAR/build"
export LD_LIBRARY_PATH="$ROCSTAR_ROOT/lib:$LD_LIBRARY_PATH"

GPU="MI250X"
rm "pubbench_results_${GPU}.csv"
touch "pubbench_results_${GPU}.csv"
opTypes=("add" "mul" "muladd" "div" "rsqrt")
dataTypes=("int8" "int16" "int32" "int64" "fp16" "fp32" "fp64")
EXP=10

mkdir -p "./rocstar_metrics/$GPU"

for nOp in "${nOps[@]}"; do
    echo "Running PubBench with $nOp ops"
    EXP=10
    make amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP
    ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --muladd --fp32
    AVG_TIME=$(cat meanDuration.txt)
    BASE_EXP=$(echo "$AVG_TIME" | awk '{ x = int((20000 / $1) + 0.5); if (x > 10) print x; else print 10 }')
    echo "New exp: $BASE_EXP"
    make amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$BASE_EXP

    for opType in "${opTypes[@]}"; do
        for dataType in "${dataTypes[@]}"; do
            export ROCSTAR_OUTPUT_FILE="./rocstar_metrics/$GPU/hw_metrics_${nOp}_${opType}_${dataType}.csv"
            EXP=$BASE_EXP
            if [[ "$dataType" == "int64" || "$opType" == "div" || "$opType" == "rsqrt" ]]; then
            # echo "nOps: ${nOp}, opType: ${opType}, dataType: ${dataType}"
                ./benchWarmer-amd_${GPU}_${nOp}_10 -u "$GPU" --"$opType" --"$dataType"
                AVG_TIME=$(cat meanDuration.txt)
                EXP=$(echo "$AVG_TIME" | awk '{ x = int((20000 / $1) + 0.5); if (x > 10) print x; else print 10 }')
                make amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP
            fi
            ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --"$opType" --"$dataType"  --rocstar --record
        done
    done
done