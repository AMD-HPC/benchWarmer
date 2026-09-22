// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "benchWarmer_run_operations.cuh"
#include "benchWarmer_api.h"

void run_int8_benchmarks(
		bool add, bool mul, bool muladd, bool div, bool shift,
		bool mfma,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
	run_operations<uint8_t>(add, mul, muladd, div, false, shift, mfma,
			grid_size, block_size, buffer_size, experiments, arch);
}
