#ifndef BENCHWARMER_RUN_OPERATIONS_VALU_CUH
#define BENCHWARMER_RUN_OPERATIONS_VALU_CUH

#include "benchWarmer_kernels.cuh"

#include <string>
#include <type_traits>
#include <unordered_map>

// Initialize buffer with random values (per element or per byte) or zero-fill.
template<class T>
static void init_gpu_buffer(T *ptr, int count, size_t byte_len, unsigned long long seed) {
	if (!g_random_buffer_init) {
		ASSERT(hipMemset((void *)ptr, 0, byte_len));
	} else {
		const int blockSizeRandom = 256;
		int gridSizeRandom = (count + blockSizeRandom - 1) / blockSizeRandom;
		initializeRandom<T><<<gridSizeRandom, blockSizeRandom>>>(ptr, count, seed);
	}
	ASSERT(hipDeviceSynchronize());
}

// Common name of a datatype (e.g. "fp32" for float), unless overridden
template<class T>
static std::string datatype_name(const char* datatype_override = nullptr) {
	if (datatype_override != nullptr) {
		return datatype_override;
	}

	// Map to get common names of datatypes
	std::unordered_map<std::string, std::string> typeNameLookup = {
			{typeid(uint32_t).name(), "int32"},
			{typeid(uint64_t).name(), "int64"},
			{typeid(float).name(), "fp32"},
			{typeid(double).name(), "fp64"},
			{typeid(__half).name(), "fp16"},
			{typeid(uint16_t).name(), "int16"},
			{typeid(uint8_t).name(), "int8"}
#if __HIPCC__
			,{typeid(__hip_fp8_storage_t).name(), "fp8"}
			,{typeid(hip_bfloat16).name(), "bf16"}
			,{typeid(float2).name(), "fp32x2"}
#elif __CUDACC__
			,{typeid(__nv_bfloat16).name(), "bf16"}
			,{typeid(float2).name(), "fp32x2"}
#endif
	};
	return typeNameLookup.find(typeid(T).name())->second;
}

template<class T>
static void run_operations_valu(void (*kernel)(T *, uint32_t), const char *op_name, int grid_size, int block_size, int buffer_size, int numExperiments, GPUArch arch, const char* datatype_override = nullptr) {

	(void)arch;
	uint64_t nThreads = (uint64_t)grid_size * (uint64_t)block_size;
	int nSize = buffer_size/sizeof(T);  // total number of elements

	uint64_t totalFlops = (uint64_t)nSize  * (uint64_t)nOps;
	// Each element in memory gets one read and one write
	uint64_t totalBytes = (uint64_t)nSize * (uint64_t)sizeof(T) * 2;

	std::string datatype = datatype_name<T>(datatype_override);

	std::string op = op_name;

	// Threads past the end of the buffer exit without doing any work, so the
	// measured throughput would not reflect the requested launch configuration
	if (nThreads > (uint64_t)std::max(nSize, 0)) {
		fprintf(stderr,
				"Error: skipping %s %s: %lu threads (grid-size %d x block-size %d) exceed %d buffer elements. "
				"Increase --buffer-size to at least %lu bytes or reduce --grid-size/--block-size.\n",
				datatype.c_str(), op.c_str(), nThreads, grid_size, block_size, nSize,
				nThreads * (uint64_t)sizeof(T));
		return;
	}

	// Double flop count for MulAdd tests since MulAdd involves two operations, multiply and add
	if(op == "MulAdd") {
		totalFlops *= 2;
	}
	if(datatype == "fp32x2") {
		totalFlops *= 2;
	}

	T *memBlock;
	ASSERT(hipMalloc((void**)&memBlock, buffer_size));

	unsigned long long seed = 12345;

	// Randomize buffer (or zero-fill) for every test except Mul, which uses typed
	// random to limit range and avoid overflow.
	if (op == "Mul") {
		init_gpu_buffer<T>(memBlock, nSize, (size_t)buffer_size, seed);
	} else {
		init_gpu_buffer<uint8_t>((uint8_t *)memBlock, buffer_size, (size_t)buffer_size, seed);
	}

	// WARMUP KERNEL
	kernel<<<dim3(grid_size), dim3(block_size)>>>((T *)memBlock, nSize);
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
		if (op == "Mul") {
			init_gpu_buffer<T>(memBlock, nSize, (size_t)buffer_size, seed + n);
		} else {
			init_gpu_buffer<uint8_t>((uint8_t *)memBlock, buffer_size, (size_t)buffer_size, seed + n);
		}

		initTimeEvents(start, stop);
		kernel<<<dim3(grid_size), dim3(block_size)>>>((T *)memBlock, nSize);
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
	printf("%s, %s, %f, %.3f, %.3f, %lu, %lu, %.3f, %d, %lu, %d\n", datatype.c_str(), op.c_str(), meanThroughput, stdevThroughput, meanDuration, totalFlops, totalBytes, ((float)totalFlops/(float)totalBytes), grid_size, nThreads, numExperiments);

	// Clean up time
	ASSERT(hipFree(memBlock));
	free(throughputs);
	free(durations);
}

#endif // BENCHWARMER_RUN_OPERATIONS_VALU_CUH
