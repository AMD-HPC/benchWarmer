#!/bin/bash
#SBATCH -p caldera
#SBATCH -w TheraC98
#SBATCH --gpus-per-node=1
#SBATCH --output=./slurm_output/h100.out
#SBATCH --error=./slurm_output/h100.err

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"
# echo "$SCRIPT_DIR"
cd "/home/khoffmey/work/PubBench"
module load CUDA

GPU="H100"
rm pubbench_results_H100.csv
touch pubbench_results_H100.csv
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
    EXP=$(echo "(3000 / $AVG_TIME + 0.5)/1" | bc)
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



# cp /home/khoffmey/work/PubBench/pubbench_results.csv "/home/khoffmey/work/PubBench/pubbench_results_backup.csv"
# mv /home/khoffmey/work/PubBench/pubbench_results_backup.csv "/home/khoffmey/work/gpu_power_frequency_study/kernels/results/pubbench_results.csv"

# cd "/home/khoffmey/work/gpu_power_frequency_study"
# source env/bin/activate

# python3 rooflinePlotterPlotly.py -e "./kernels/results/pubbench_results.csv" -g "MI250X" -d "INT8 INT16 INT32 INT64 FP16 FP32 FP64" -o "ADD MULADD MUL DIV RSQ" -m "HBM" -c 1