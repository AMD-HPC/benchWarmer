#!/bin/bash
# cd /home/khoffmey/work/PubBench
mkdir -p runtime_data
cat pubbench_results_A100.csv > pubbench_results_all_gpus.csv
tail -n +2 pubbench_results_MI250X.csv >> pubbench_results_all_gpus.csv
tail -n +2 pubbench_results_MI300A.csv >> pubbench_results_all_gpus.csv
tail -n +2 pubbench_results_MI300X.csv >> pubbench_results_all_gpus.csv
tail -n +2 pubbench_results_H100.csv >> pubbench_results_all_gpus.csv

mv pubbench_results_all_gpus.csv /home/khoffmey/work/PubBench/runtime_data/pubbench_results_all_gpus.csv
echo "PubBench results combined into pubbench_results_all_gpus.csv"