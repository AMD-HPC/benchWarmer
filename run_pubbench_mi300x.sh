#!/bin/bash
#SBATCH -p MI300x
#SBATCH -w TheraC18
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=./slurm_output/mi300x.out
#SBATCH --error=./slurm_output/mi300x.err

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR" 
# echo "$SCRIPT_DIR"
module load rocm

# cd "/home/khoffmey/work/PubBench"
# pwd
# ls

nOps=()
# nOps=2
for ((i=0; i<16; i++)); do
    nOp=$(echo "2^$i" | bc)
    nOps+=($nOp)
done

ROCSTAR_ROOT="../rocSTAR/build"
export LD_LIBRARY_PATH="$ROCSTAR_ROOT/lib:$LD_LIBRARY_PATH"
# AGT_PATH="../agt_files/agt_internal"

GPU="MI300X"
rm "pubbench_results_${GPU}.csv"
touch "pubbench_results_${GPU}.csv"
opTypes=("add" "mul" "muladd" "div" "rsqrt")
dataTypes=("int8" "int16" "int32" "int64" "fp16" "fp32" "fp64")

mkdir -p "./rocstar_metrics/$GPU"
# mkdir -p "./agt_metrics/$GPU"

for nOp in "${nOps[@]}"; do
    echo "Running PubBench with $nOp ops"
    # make clean
    EXP=10
    make -B amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP
    ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --muladd --fp32
    AVG_TIME=$(cat meanDuration.txt)
    EXP=$(echo "(2000 / $AVG_TIME + 0.5)/1" | bc)
    echo "New exp: $EXP"
    make -B amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP

    for opType in "${opTypes[@]}"; do
        for dataType in "${dataTypes[@]}"; do
            # EXP=5
            # echo "nOps: ${nOp}, opType: ${opType}, dataType: ${dataType}"
            # ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --"$opType" --"$dataType"

            # AVG_TIME=$(cat meanDuration.txt)
            # EXP=$(echo "(1500 / $AVG_TIME + 0.5)/1" | bc)
            # echo "New exp: $EXP"
            # ((EXP=17/AVG_TIME))
            
            # if AI >
            # nOp=8
            # opType="muladd"
            # dataType="int64"
            export ROCSTAR_OUTPUT_FILE="./rocstar_metrics/$GPU/hw_metrics_${nOp}_${opType}_${dataType}.csv"
            # export AGT_OUTPUT_FILE="./agt_metrics/$GPU/hw_metrics_${nOp}_${opType}_${dataType}.csv"
            # AGT_CMD="sudo $AGT_PATH -unilog=PM -unilogallgroups -i=0,1,2,3,4,5,6,7 -unilogperiod=50 -unilogoutput=${AGT_OUTPUT_FILE} &"
            # eval $AGT_CMD
            # echo "Starting AGT: $(date)"
            # make -B amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP
            ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --"$opType" --"$dataType" --record --rocstar

            # NVIDIA_OUTPUT_FILE="./nvidia_metrics/$GPU/hw_metrics_${nOp}_${opType}_${dataType}.csv"
            # ./benchWarmer-nv_${GPU}_${nOp}_10 -u "$GPU" --"$opType" --"$dataType" --record
            # nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm --format=csv -l 1 > $NVIDIA_OUTPUT_FILE & PID=$!
            # ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --"$opType" --"$dataType" 
        done
    done
done


# cp /home/khoffmey/work/PubBench/pubbench_results.csv "/home/khoffmey/work/PubBench/pubbench_results_backup.csv"
# mv /home/khoffmey/work/PubBench/pubbench_results_backup.csv "/home/khoffmey/work/gpu_power_frequency_study/kernels/results/pubbench_results.csv"

# cd "/home/khoffmey/work/gpu_power_frequency_study"
# source env/bin/activate

# python3 rooflinePlotterPlotly.py -e "./kernels/results/pubbench_results.csv" -g "MI250X" -d "INT8 INT16 INT32 INT64 FP16 FP32 FP64" -o "ADD MULADD MUL DIV RSQ" -m "HBM" -c 1