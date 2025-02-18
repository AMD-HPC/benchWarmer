#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
make amd nOps=1000
./pubBench-amd --muladd

mv pubbench_results.csv "$SCRIPT_DIR/../gpu_power_frequency_study/kernels/results/pubbench_results.csv"