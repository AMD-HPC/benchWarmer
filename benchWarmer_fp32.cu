// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "benchWarmer_run_operations.cuh"
#include "benchWarmer_api.h"

void run_fp32_benchmarks(
		bool add, bool mul, bool muladd, bool div, bool rsq, bool mfma,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
	run_operations<float>(add, mul, muladd, div, rsq, false, mfma,
			grid_size, block_size, buffer_size, experiments, arch);
}
