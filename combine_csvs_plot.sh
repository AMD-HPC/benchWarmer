#!/bin/bash

./combine_csvs.sh

python3 interactivePlotter.py -r "./runtime_data/pubbench_results_all_gpus.csv" -g "ALL" -d "INT8 INT16 INT32 INT64 FP16 FP32 FP64" -o "ADD MULADD MUL DIV RSQ" -m "HBM"