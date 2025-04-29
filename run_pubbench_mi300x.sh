#!/bin/bash
#SBATCH -p MI300x
#SBATCH -w TheraC18
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=./slurm_output/mi300x.out
#SBATCH --error=./slurm_output/mi300x.err

module load rocm

nOps=()
# nOps=2
for ((i=0; i<16; i++)); do
    nOp=$(echo "2^$i" | bc)
    nOps+=($nOp)
done

ROCSTAR_ROOT="../rocSTAR/build"
export LD_LIBRARY_PATH="$ROCSTAR_ROOT/lib:$LD_LIBRARY_PATH"

GPU="MI300X"
rm "pubbench_results_${GPU}.csv"
touch "pubbench_results_${GPU}.csv"
opTypes=("add" "mul" "muladd" "div" "rsqrt")
dataTypes=("int8" "int16" "int32" "int64" "fp16" "fp32" "fp64")

mkdir -p "./rocstar_metrics/$GPU"

for nOp in "${nOps[@]}"; do
    echo "Running PubBench with $nOp ops"
    # make clean
    EXP=10
    make -B amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP
    ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --muladd --fp32
    AVG_TIME=$(cat meanDuration.txt)
    EXP=$(echo "$AVG_TIME" | awk '{ x = int((3000 / $1) + 0.5); if (x > 10) print x; else print 10 }')
    echo "New exp: $EXP"
    make -B amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP

    for opType in "${opTypes[@]}"; do
        for dataType in "${dataTypes[@]}"; do
            export ROCSTAR_OUTPUT_FILE="./rocstar_metrics/$GPU/hw_metrics_${nOp}_${opType}_${dataType}.csv"
            ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --"$opType" --"$dataType" --record --rocstar
        done
    done
done