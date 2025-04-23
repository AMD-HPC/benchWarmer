#!/bin/bash
##SBATCH -w x1000c3s5b1n0
##SBATCH -w x1000c2s5b1n0
#SBATCH -w x1000c3s1b1n0
#SBATCH --gpus-per-node=1
#SBATCH --time=36:00:00
#SBATCH --output=./slurm_output/mi250x.out
#SBATCH --error=./slurm_output/mi250x.err

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"
# echo "$SCRIPT_DIR"
# cd "/home/khoffmey/work/PubBench"
# pwd
# ls
# module load rocm/6.1.1
module load rocm

nOps=()
# nOps=2
for ((i=0; i<16; i++)); do
    nOp=$(echo "2^$i" | bc)
    nOps+=($nOp)
done

ROCSTAR_ROOT="../rocSTAR/build"
export LD_LIBRARY_PATH="$ROCSTAR_ROOT/lib:$LD_LIBRARY_PATH"
AGT_PATH="../agt_files/agt_internal"

GPU="MI250X"
rm "pubbench_results_${GPU}.csv"
touch "pubbench_results_${GPU}.csv"
opTypes=("add" "mul" "muladd" "div" "rsqrt")
dataTypes=("int8" "int16" "int32" "int64" "fp16" "fp32" "fp64")
EXP=10

mkdir -p "./rocstar_metrics/$GPU"
# mkdir -p "./agt_metrics/$GPU"

for nOp in "${nOps[@]}"; do
    echo "Running PubBench with $nOp ops"
    # make clean
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
            # EXP=$(echo "$AVG_TIME" | awk '{ x = int((20000 / $1) + 0.5); if (x > 10) print x; else print 10 }')
            # echo "New exp: $EXP"
            # ((EXP=17/AVG_TIME))
            
            # if AI >
            # nOp=8
            # opType="muladd"
            # dataType="int64"
            # export ROCSTAR_OUTPUT_FILE="./rocstar_metrics/$GPU/hw_metrics_${nOp}_${opType}_${dataType}.csv"
            # export AGT_OUTPUT_FILE="./agt_metrics/$GPU/hw_metrics_${nOp}_${opType}_${dataType}.csv"
            # AGT_CMD="sudo $AGT_PATH -unilog=PM -unilogallgroups -i=0,1,2,3,4,5,6,7 -unilogperiod=50 -unilogoutput=${AGT_OUTPUT_FILE} &"
            # eval $AGT_CMD
            # echo "Starting AGT: $(date)"
            # make -B amd nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP
            ./benchWarmer-amd_${GPU}_${nOp}_${EXP} -u "$GPU" --"$opType" --"$dataType"  --rocstar --record
        done
    done
done

# cp /home/khoffmey/work/PubBench/pubbench_results.csv "/home/khoffmey/work/PubBench/pubbench_results_backup.csv"
# mv /home/khoffmey/work/PubBench/pubbench_results_backup.csv "/home/khoffmey/work/gpu_power_frequency_study/kernels/results/pubbench_results.csv"

# cd "/home/khoffmey/work/gpu_power_frequency_study"
# source env/bin/activate

# python3 rooflinePlotterPlotly.py -e "./kernels/results/pubbench_results.csv" -g "MI250X" -d "INT8 INT16 INT32 INT64 FP16 FP32 FP64" -o "ADD MULADD MUL DIV RSQ" -m "HBM" -c 1