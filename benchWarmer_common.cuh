#ifndef BENCHWARMER_COMMON_CUH
#define BENCHWARMER_COMMON_CUH

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

#if __HIPCC__

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>  // for __half
#include <hip/hip_fp8.h>
#include <hip/hip_bfloat16.h>
#include <hiprand/hiprand_kernel.h>

#elif __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // for __half
#include <cuda_bf16.h>  // for __nv_bfloat16
#include <curand_kernel.h>

// The code is written against the HIP API; map every HIP name it uses to CUDA
#define hipSuccess cudaSuccess
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemset cudaMemset
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventDestroy cudaEventDestroy

#define hiprandState curandState
#define hiprand_init curand_init
#define hiprand_uniform curand_uniform
#define hiprand curand

#endif

#include <algorithm>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Number of instructions per read/write 
// Must be set at compile time due to loop unrolling
#ifndef nOps
#define nOps 10000
#endif

#define DEFAULT_WORKGROUP_SIZE 256
#define DEFAULT_WORKGROUPS 14592*10
#define DEFAULT_NUM_EXPERIMENTS 10
#define DEFAULT_DATASET_SIZE 1024 * 1024 * 1024

/* Long-option return value for getopt_long_only (not a short option char). */
#define OPT_NO_RANDOM_BUFFER_INIT 300
#define OPT_FP32X2 301

#define ASSERT(x) (assert((x)==hipSuccess))

extern bool g_random_buffer_init;


// Runtime GPU architecture detection
enum class GPUArch {
	GFX908,      // MI100
	GFX90A,      // MI200
	GFX940,      // MI300A/MI300X (gfx940, gfx941, gfx942)
	GFX950,      // MI350X/MI355X (gfx950)
	UNKNOWN
};

inline GPUArch getGPUArchitecture() {
	#if __HIPCC__
	hipDeviceProp_t props;
	ASSERT(hipGetDeviceProperties(&props, 0));
	std::string gcnArchName(props.gcnArchName);

	if (gcnArchName.find("gfx908") != std::string::npos) {
		return GPUArch::GFX908;
	} else if (gcnArchName.find("gfx90a") != std::string::npos) {
		return GPUArch::GFX90A;
	} else if (gcnArchName.find("gfx940") != std::string::npos ||
	           gcnArchName.find("gfx941") != std::string::npos ||
	           gcnArchName.find("gfx942") != std::string::npos) {
		return GPUArch::GFX940;
	} else if (gcnArchName.find("gfx950") != std::string::npos) {
		return GPUArch::GFX950;
	}
	#endif
	return GPUArch::UNKNOWN;
}

static inline
void initTimeEvents(hipEvent_t &start, hipEvent_t &stop)
{
	ASSERT(hipEventCreate(&start));
	ASSERT(hipEventCreate(&stop));
	ASSERT(hipEventRecord(start));
}

static inline
void stopTimeEvents(float &eventMs, hipEvent_t &start, hipEvent_t &stop)
{
	ASSERT(hipEventRecord(stop));
	ASSERT(hipEventSynchronize(stop));
	ASSERT(hipEventElapsedTime(&eventMs, start, stop));
	ASSERT(hipEventDestroy(start));
	ASSERT(hipEventDestroy(stop));
}

inline void stats(float *samples, int entries, float *mean, float *stdev, float *confidence)
{
	float mean_val, stdev_val;

	mean_val = 0;
	for(int i=0; i<entries; i++)
	{
		mean_val += samples[i];
	}
	mean_val = mean_val / entries;

	stdev_val = 0;
	for(int i=0; i<entries; i++)
	{
		stdev_val += (samples[i] - mean_val) * (samples[i] - mean_val);
	}
	stdev_val = sqrtf(stdev_val / entries);

	// return
	mean[0] = mean_val;
	stdev[0] = stdev_val;
	confidence[0] = 1.960 * stdev_val / sqrtf(entries);
}

#endif // BENCHWARMER_COMMON_CUH
