// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "benchWarmer_api.h"
#if __HIPCC__
#include "benchWarmer_run_operations_mfma.cuh"
#endif

void run_fp8_benchmarks(
		bool mfma,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
#if __HIPCC__
	// FP8 is requested by name since __hip_fp8_storage_t is typedef'd to uint8_t
	// and would otherwise be treated as int8. Only gfx940 and gfx950 have FP8 MFMA.
	if (mfma) {
		run_operations_mfma("fp8", grid_size, block_size, buffer_size, experiments, arch);
	}
#else
	(void)mfma;
	(void)grid_size;
	(void)block_size;
	(void)buffer_size;
	(void)experiments;
	(void)arch;
#endif
}
