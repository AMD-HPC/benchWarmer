// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "benchWarmer_run_operations.cuh"
#include "benchWarmer_api.h"

void run_fp32x2_benchmarks(
		bool add, bool mul, bool muladd, bool div, bool rsq,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
	run_operations_not_mfma<float2>(add, mul, muladd, div, rsq, false,
			grid_size, block_size, buffer_size, experiments, arch);
}
