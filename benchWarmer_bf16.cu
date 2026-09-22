// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "benchWarmer_run_operations.cuh"
#include "benchWarmer_api.h"

void run_bf16_benchmarks(
		bool add, bool mul, bool muladd, bool div, bool rsq, bool mfma,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
#if __HIPCC__
	run_operations<hip_bfloat16>(add, mul, muladd, div, rsq, false, mfma,
			grid_size, block_size, buffer_size, experiments, arch);
#elif __CUDACC__
	run_operations<__nv_bfloat16>(add, mul, muladd, div, rsq, false, mfma,
			grid_size, block_size, buffer_size, experiments, arch);
#endif
}
