#!/bin/bash
#SBATCH -p caldera
#SBATCH -w TheraC98
#SBATCH --gpus-per-node=1
#SBATCH --output=./slurm_output/h100.out
#SBATCH --error=./slurm_output/h100.err

module load CUDA

GPU="H100"
# rm pubbench_results_H100.csv
# touch pubbench_results_H100.csv
opTypes=("add" "mul" "muladd" "div" "rsqrt")
dataTypes=("int8" "int16" "int32" "int64" "fp16" "fp32" "fp64")

nOps=()
# nOps=2
for ((i=0; i<16; i++)); do
    nOp=$(echo "2^$i" | bc)
    nOps+=($nOp)
done

EXP=100
mkdir -p "nvidia_metrics/$GPU"
for nOp in "${nOps[@]}"; do
    EXP=10
    make nv nOps=$nOp GPU=$GPU EXP=$EXP
    ./benchWarmer-nv_${GPU}_${nOp}_${EXP} -u "$GPU" --muladd --fp32
    AVG_TIME=$(cat meanDuration.txt)
    EXP=$(echo "$AVG_TIME" | awk '{ x = int((3000 / $1) + 0.5); if (x > 10) print x; else print 10 }')
    echo "New exp: $EXP"
    make nv nOps=$nOp GPU=$GPU EXP=$EXP
    for opType in "${opTypes[@]}"; do
        for dataType in "${dataTypes[@]}"; do
            echo "Running PubBench with $nOp ops"
            # make clean
            echo "Running $opType and $dataType"
            NVIDIA_OUTPUT_FILE="./nvidia_metrics/$GPU/hw_metrics_${nOp}_${opType}_${dataType}.csv"
            ./benchWarmer-nv_${GPU}_${nOp}_10 -u "$GPU" --"$opType" --"$dataType" --record

            nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm --format=csv -lms 10 > $NVIDIA_OUTPUT_FILE & PID=$!
        
            timeout 5s ./benchWarmer-nv_${GPU}_${nOp}_${EXP} -u "$GPU" --"$opType" --"$dataType"
            
            kill $PID
        done
    done
done