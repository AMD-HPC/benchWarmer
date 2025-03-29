#!/bin/bash
#SBATCH -p caldera
#SBATCH -w TheraC98
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=/home/khoffmey/work/gpu_power_frequency_study/slurm_output/pubbench_job.out
#SBATCH --error=/home/khoffmey/work/gpu_power_frequency_study/slurm_output/pubbench_job.err

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"
# echo "$SCRIPT_DIR"
cd "/home/khoffmey/work/PubBench"
module load CUDA

GPU="H100"
rm "pubbench_results_${GPU}.csv"
touch "pubbench_results_${GPU}.csv"
# pwd
# ls

nOps=()
# nOps=2
for ((i=1; i<16; i++)); do
    nOp=$(echo "2^$i" | bc)
    nOps+=($nOp)
done

EXP=10
for nOp in "${nOps[@]}"; do
    echo "Running PubBench with $nOp ops"
    # make clean
    make nv nOps=$nOp GPU=$GPU ROCSTAR_ROOT=$ROCSTAR_ROOT EXP=$EXP
    echo "Running benchWarmer-nv_$nOp -u $GPU"
    ./benchWarmer-nv_${GPU}_${nOp}_${EXP} -u "$GPU" --add --mul --muladd --div --rsqrt --record
done




# cp /home/khoffmey/work/PubBench/pubbench_results.csv "/home/khoffmey/work/PubBench/pubbench_results_backup.csv"
# mv /home/khoffmey/work/PubBench/pubbench_results_backup.csv "/home/khoffmey/work/gpu_power_frequency_study/kernels/results/pubbench_results.csv"

# cd "/home/khoffmey/work/gpu_power_frequency_study"
# source env/bin/activate

# python3 rooflinePlotterPlotly.py -e "./kernels/results/pubbench_results.csv" -g "MI250X" -d "INT8 INT16 INT32 INT64 FP16 FP32 FP64" -o "ADD MULADD MUL DIV RSQ" -m "HBM" -c 1