#ifndef BENCHWARMER_RUN_OPERATIONS_MFMA_CUH
#define BENCHWARMER_RUN_OPERATIONS_MFMA_CUH

#include "benchWarmer_run_operations_valu.cuh"
#if __HIPCC__
#include "benchWarmer_mfma.cuh"
#endif

#if __HIPCC__
// MFMA benchmarking function - AMD-specific
// Does nothing if the architecture has no MFMA instruction for the datatype
static inline void run_operations_mfma(const std::string &datatype, int grid_size, int block_size, int buffer_size, int numExperiments, GPUArch arch) {

	mfma_kernel_t kernel = select_mfma_kernel(datatype, arch);
	if (kernel == nullptr) {
		return;
	}

	uint64_t nThreads = (uint64_t)grid_size * (uint64_t)block_size;

	// Map to get total flop numbers - determine at runtime based on architecture
	std::unordered_map<std::string, int> flopFactorLookup;

	if (arch == GPUArch::GFX908) {
			// MI100
			flopFactorLookup = {
					{"int8", 16384},
					{"fp16", 16384},
					{"bf16", 8192},
					{"fp32", 4096},
					{"fp64", 2048},
			};
	} else if (arch == GPUArch::GFX90A) {
			// MI200
			flopFactorLookup = {
					{"int8", 16384},
					{"fp16", 16384},
					{"bf16", 16384},
					{"fp32", 4096},
					{"fp64", 2048},
			};
	} else if (arch == GPUArch::GFX940) {
			// MI300
			flopFactorLookup = {
					{"int8", 32768},
					{"fp8", 32768},
					{"fp16", 16384},
					{"bf16", 16384},
					{"fp32", 4096},
					{"fp64", 2048},
			};
	} else {
			// MI350 (gfx950) scale MFMA: v_mfma_f32_32x32x64_f8f6f4 -> 32*32*64*2 = 131072 ops.
			// fp8 on gfx950 also uses this instruction (mfma_f8f6f4_fp8) instead of the
			// 32x32x16_fp8_fp8 path used on gfx940.
			flopFactorLookup = {
					{"int8", 65536},
					{"fp4", 131072},
					{"fp6", 131072},
					{"fp6_e3m2", 131072},
					{"fp8", 131072},
					{"fp8_e5m2", 131072},
					{"fp16", 32768},
					{"bf16", 32768},
					{"fp32", 4096},
					{"fp64", 2048},
			};
	}

	int flopFactor = flopFactorLookup.find(datatype)->second;

	uint64_t totalFlops = (uint64_t)grid_size * 4 * nOps * flopFactor;

	// Estimate based on perf counters. Most reads/writes go to L1/L2
	uint64_t totalBytes = 512;

	float *memBlock;
	ASSERT(hipMalloc((void**)&memBlock, buffer_size));

	unsigned long long seed = 12345;
	init_gpu_buffer<uint8_t>((uint8_t *)memBlock, buffer_size, (size_t)buffer_size, seed);


	// WARMUP KERNEL
	kernel<<<grid_size, block_size>>>(memBlock);
	ASSERT(hipDeviceSynchronize());

	// Timing data
	float eventMs;
	hipEvent_t start, stop;

	// Measurement data
	float *throughputs = (float *)calloc(numExperiments, sizeof(float));
	float *durations = (float *)calloc(numExperiments, sizeof(float));

	// Run experiments
	for (int n=0; n<numExperiments; n++)
	{
		init_gpu_buffer<uint8_t>((uint8_t *)memBlock, buffer_size, (size_t)buffer_size, seed);

	initTimeEvents(start, stop);
		kernel<<<grid_size, block_size>>>(memBlock);
	stopTimeEvents(eventMs, start, stop);

		throughputs[n] = (float) totalFlops / eventMs / 1e6;  // Unit: GFLOPs/sec
		durations[n] = eventMs;
	}

	// Calculate summary statistics
	float meanThroughput, stdevThroughput, confidenceThroughput;
	float meanDuration, stdevDuration, confidenceDuration;

	stats(throughputs, numExperiments, &meanThroughput, &stdevThroughput, &confidenceThroughput);
	stats(durations, numExperiments, &meanDuration, &stdevDuration, &confidenceDuration);

	// Print output
	printf("%s, %s, %f, %.3f, %.3f, %lu, %lu, %.3f, %d, %lu, %d\n", datatype.c_str(), "mfma", meanThroughput, stdevThroughput, meanDuration, totalFlops, totalBytes, ((float)totalFlops/(float)totalBytes), grid_size, nThreads, numExperiments);

	// Clean up time
	ASSERT(hipFree(memBlock));
	free(throughputs);
	free(durations);
}
#endif // __HIPCC__

#endif
