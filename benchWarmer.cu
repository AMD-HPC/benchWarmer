#if __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // for __half

#define gpu(symbol) cuda ## symbol

#elif __HIPCC__

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>  // for __half

#define gpu(symbol) hip ## symbol

#endif

#include <string>
#include <unordered_map>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Number of computes must be set at compile time
#ifndef nOps
#define nOps 1000
#endif

#define DEFAULT_WORKGROUP_SIZE 256
#define DEFAULT_WORKGROUPS 16384
#define DEFAULT_NUM_EXPERIMENTS 10
#define DEFAULT_DATASET_SIZE 1024 * 1024 * 1024

// function classes that we want to measure

template<class T>
struct Add {
  __device__ T operator()(T x, T y, T z) {
    return x + y;
  }
};

template<class T>
struct Mul {
  __device__ T operator()(T x, T y, T z) {
    return x * y;
  }
};

template<class T>
struct MulAdd {
  __device__ T operator()(T x, T y, T z) {
    return x * y + z;
  }
};

template<class T>
struct Div {
  __device__ T operator()(T x, T y, T z) {
    return x / y;
  }
};

template<class T>
struct Rsqrt {
  __device__ T operator()(T x, T y, T z) {
    return rsqrtf(x);
  }
};

template<class T>
struct ShiftLeft {
  __device__ T operator()(T x, T y, T z) {
    return x << y;
  }
};

template<class T>
struct ShiftRight {
  __device__ T operator()(T x, T y, T z) {
    return x >> y;
  }
};

template<class T>
struct RotateLeft {
  __device__ T operator()(T x, T y, T z) {
    return (x << y) | (x >> (8 * sizeof(T) - y));
  }
};

template<class T>
struct RotateRight {
  __device__ T operator()(T x, T y, T z) {
    return (x >> y) | (x << (8 * sizeof(T) - y));
  }
};


static inline
void initTimeEvents(gpu(Event_t) &start, gpu(Event_t) &stop)
{
	assert((gpu(EventCreate(&start)))==gpu(Success));
	assert((gpu(EventCreate(&stop)))==gpu(Success));
	assert((gpu(EventRecord(start)))==gpu(Success));
}

static inline
void stopTimeEvents(float &eventMs, gpu(Event_t) &start, gpu(Event_t) &stop)
{
	assert((gpu(EventRecord(stop)) )== gpu(Success));
	assert((gpu(EventSynchronize(stop))) == gpu(Success));
	assert((gpu(EventElapsedTime(&eventMs, start, stop))) == gpu(Success));
	assert((gpu(EventDestroy(start))) == gpu(Success));
	assert((gpu(EventDestroy(stop))) == gpu(Success));
}

void stats(float *samples, int entries, float *mean, float *stdev, float *confidence)
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

template<typename T, int n, class Func>
__global__ void throughput_kernel(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;
	//const uint32_t nEntriesPerThread = (uint32_t) nSize / nThreads;

	T *a;
	a = &buf[gid];
	T x = (T)2.2;
	T y = (T)1.2;
	Func func;

	// Unroll to prevent the compiler from optimizing out the work
	#pragma unroll 1
	for(uint32_t offset=0; offset < nSize; offset += nThreads)
	{
		#pragma unroll
		for(int j=0; j<n; j++)
		{
			// Two different write locations to force the compiler to complete every operation
			x = func(a[offset], x, y);
		}
	}
  a[0] = x;
}

// This kernel is similar to throughput_kernel, but it employs a workaround to prevent
// the compiler from optimizing out the work for some datatypes/instructions
template<typename T, int n, class Func>
__global__ void throughput_kernel_unrolled(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;
	//const uint32_t nEntriesPerThread = (uint32_t) nSize / nThreads;

	T *a;
	a = &buf[gid];
	T x = (T)2.2;
	T y = (T)1.2;
	Func func;

	// Unroll to prevent the compiler from optimizing out the work
	#pragma unroll 1
	for(uint32_t offset=0; offset < nSize; offset += nThreads)
	{
		#pragma unroll
		for(int j=0; j<n; j+=2)
		{
			// Two different write locations to force the compiler to complete every operation
			x = func(a[offset], x, y);
			a[offset] = func(a[offset], x, y);
		}
	}
}

// MI200 hardware uses packed arithmetic on FP32 Add, Multiply, and FMA instructions
template<int n, class Func>
__global__ void packed_throughput_kernel(float2 *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;
	//const uint32_t nEntriesPerThread = (uint32_t) nSize / nThreads;

	float2 *a;
	a = &buf[gid];
	float2 x = {2.2f,2.2f};
	float2 y = {1.2f,1.2f};
	Func func;

	// Unroll to prevent the compiler from optimizing out the work
	for(uint32_t offset=0; offset < nSize; offset += nThreads)
	{
		for(int j=0; j<n; j++)
		{
			// Two different write locations to force the compiler to complete every operation
			x = {func(a[offset].x, x.x, y.x), func(a[offset].y, x.y, y.y)};
		}
	}
	a[0] = x;
}

template<class T, class Func>
static void bench_func(void) {
  
  void *memBlock;
  int numWorkgroups = DEFAULT_WORKGROUPS;
  int workgroupSize = DEFAULT_WORKGROUP_SIZE;
  int numExperiments = DEFAULT_NUM_EXPERIMENTS;

  uint64_t nThreads = (uint64_t)numWorkgroups * (uint64_t)workgroupSize;
  int nSize = DEFAULT_DATASET_SIZE/sizeof(T);  // total number of ints/floats
  uint64_t totalFlops = (uint64_t)nSize  * (uint64_t)nOps;

  // Map for common names of datatypes
  std::unordered_map<std::string, std::string> typeNameLookup = {
      {typeid(uint32_t).name(), "int32"},
      {typeid(uint64_t).name(), "int64"},
      {typeid(float).name(), "fp32"},
      {typeid(double).name(), "fp64"},
      {typeid(__half).name(), "fp16"},
      {typeid(uint16_t).name(), "int16"},
      {typeid(uint8_t).name(), "int8"}
  };
  std::string datatype = typeNameLookup.find(typeid(T).name())->second;

  // s = the name of the operation (after some processing)
	std::string op = typeid(Func).name();
  // Remove all numbers (digits)
  op.erase(std::remove_if(op.begin(), op.end(), ::isdigit), op.end());

  // Find the first occurrence of 'I' and truncate the string at that point
  size_t pos = op.find('I');
  if (pos != std::string::npos) {
      op = op.substr(0, pos);  // Truncate the string at 'I'
  }
	// Double flop count for MulAdd tests since MulAdd involves two operations, multiply and add
	if(op == "MulAdd") {
		totalFlops *= 2;
	}
  uint64_t totalBytes = (uint64_t)nSize * (uint64_t)sizeof(T);

  assert((gpu(Malloc(&memBlock, DEFAULT_DATASET_SIZE)))==gpu(Success));
	

  // WARMUP KERNEL
  // packed_throughput_kernel: FP32 Add, Mul, MulAdd
  // throughput_kernel_unrolled: Rsqrt, All Integer Add, Mul
  // throughput_kernel: All other tests
  if (datatype == "fp32" && (op == "MulAdd")) {
    // FP32 MulAdd
    packed_throughput_kernel<nOps,MulAdd<float>><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((float2 *)memBlock, nSize/2);
  } else if (datatype == "fp32" && (op == "Add")) {
    // FP32 Add
    packed_throughput_kernel<nOps,Add<float>><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((float2 *)memBlock, nSize/2);
  } else if (datatype == "fp32" && (op == "Mul")) {
    // FP32 Mul
    packed_throughput_kernel<nOps,Mul<float>><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((float2 *)memBlock, nSize/2);
  } else if ((datatype.find("int") != std::string::npos && (op == "Add" || op == "Mul")) || op == "Rsqrt") {
    // Rsqrt, Integer Add, Mul
    throughput_kernel_unrolled<T,nOps,Func><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((T *)memBlock, nSize);
  } else {
    // Every other test
    throughput_kernel<T,nOps,Func><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((T *)memBlock, nSize);
  }
  gpu(DeviceSynchronize());

	// Timing data
  float eventMs;
  gpu(Event_t) start, stop;

  // Measurement data 
  float *throughputs = (float *)calloc(numExperiments, sizeof(float));
  float *durations = (float *)calloc(numExperiments, sizeof(float));

	// Run experiments
  for (int n=0; n<numExperiments; n++)
  {
		// packed_throughput_kernel: FP32 Add, Mul, MulAdd
		// throughput_kernel_unrolled: Rsqrt, All Integer Add, Mul
		// throughput_kernel: All other tests
    if (datatype == "fp32" && (op == "MulAdd")) {
      // FP32 MulAdd
			initTimeEvents(start, stop);
      packed_throughput_kernel<nOps,MulAdd<float>><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((float2 *)memBlock, nSize/2);
			stopTimeEvents(eventMs, start, stop);
    } else if (datatype == "fp32" && (op == "Add")) {
      // FP32 Add
			initTimeEvents(start, stop);
      packed_throughput_kernel<nOps,Add<float>><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((float2 *)memBlock, nSize/2);
			stopTimeEvents(eventMs, start, stop);
    } else if (datatype == "fp32" && (op == "Mul")) {
      // FP32 Mul
			initTimeEvents(start, stop);
      packed_throughput_kernel<nOps,Mul<float>><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((float2 *)memBlock, nSize/2);
			stopTimeEvents(eventMs, start, stop);
    } else if ((datatype.find("int") != std::string::npos && (op == "Add" || op == "Mul")) || op == "Rsqrt") {
      // Rsqrt, Integer Add, Mul
			initTimeEvents(start, stop);
      throughput_kernel_unrolled<T,nOps,Func><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((T *)memBlock, nSize);
			stopTimeEvents(eventMs, start, stop);
    } else {
      // Every other test
			initTimeEvents(start, stop);
      throughput_kernel<T,nOps,Func><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((T *)memBlock, nSize);
			stopTimeEvents(eventMs, start, stop);
    }

    throughputs[n] = (float) totalFlops / eventMs / 1e6;  // Unit: GFLOPs/sec
    durations[n] = eventMs;
  }

  // Calculate summary statistics
  float meanThroughput, stdevThroughput, confidenceThroughput;
  float meanDuration, stdevDuration, confidenceDuration;

  stats(throughputs, numExperiments, &meanThroughput, &stdevThroughput, &confidenceThroughput);
  stats(durations, numExperiments, &meanDuration, &stdevDuration, &confidenceDuration);

  printf("%s, %s, %f, %.3f, %.3f, %lu, %lu, %.3f, %d, %lu, %d\n", datatype.c_str(), op.c_str(), meanThroughput, stdevThroughput, meanDuration, totalFlops, totalBytes, ((float)totalFlops/(float)totalBytes), numWorkgroups, nThreads, numExperiments);

	// Print output
  /*
  printf("workgroupSize:%d, workgroups:%d, nThreads: %lu, nSize: %d, experiments: %d\n",
      workgroupSize, numWorkgroups, nThreads, nSize, numExperiments);
  printf("    Total FLOPS=%lu, total bytes accessed=%lu, AI=%f, mean duration=%.3f ms\n\n",
      totalFlops, totalBytes, ((float)totalFlops/(float)totalBytes), meanDuration);
  printf("    Mean throughput=%f GFLOPs/sec, stdev=%.3f GFLOPs/s, 95%% Confidence Interval: [%.3f, %.3f]\n\n",
      meanThroughput, stdevThroughput, meanThroughput - confidenceThroughput, meanThroughput + confidenceThroughput);
      */

  // Clean up time
  gpu(Free(memBlock));
}

// Run relevent tests for integer types
template<class T>
static void bench_int(bool add, bool mul, bool muladd, bool div, bool rsq, bool shift, bool rotate) {

	if(add) {
		bench_func<T,Add<T>>();
	}
	if(mul) {
		bench_func<T,Mul<T>>();
	}
	if(muladd) {
		bench_func<T,MulAdd<T>>();
	}
	if(div) {
		bench_func<T,Div<T>>();
	}
	if(rsq) {
		bench_func<T,Rsqrt<T>>();
	}
	if(shift) {
		bench_func<T,ShiftLeft<T>>();
		bench_func<T,ShiftRight<T>>();
	}
	if(rotate) {
		bench_func<T,RotateLeft<T>>();
		bench_func<T,RotateRight<T>>();
	}
}


// Run relevent tests for floating point types
template<class T>
static void bench_fp(bool add, bool mul, bool muladd, bool div, bool rsq) {
	if(add) {
		bench_func<T,Add<T>>();
	}
	if(mul) {
		bench_func<T,Mul<T>>();
	}
	if(muladd) {
		bench_func<T,MulAdd<T>>();
	}
	if(div) {
		bench_func<T,Div<T>>();
	}
	if(rsq) {
		bench_func<T,Rsqrt<T>>();
	}
}

int main(int argc, char **argv)
{

	//CLI parsing
	bool i8 = false, i16 = false, i32 = false, i64 = false, fp16 = false, fp32 = false, fp64 = false;
	bool add = false, mul = false, muladd = false, div = false, rsq = false, shift = false, rotate = false;

	int c, option_index = 0;
	static struct option long_options[] = {
					{"h",       no_argument,   0,  'h' },
					{"a",       no_argument,   0,  'a' },
					{"int8",    no_argument,   0,  'g' },
					{"int16",   no_argument,   0,  'b' },
					{"int32",   no_argument,   0,  'c' },
					{"int64",   no_argument,   0,  'd' },
					{"fp16",    no_argument,   0,  'v' },
					{"fp32",    no_argument,   0,  'e' },
					{"fp64",    no_argument,   0,  'f' },
					{"int",     no_argument,   0,  'i' },
					{"fp",      no_argument,   0,  'j' },
					{"add",     no_argument,   0,  'k' },
					{"mul",     no_argument,   0,  'l' },
					{"muladd",  no_argument,   0,  'm' },
					{"div",     no_argument,   0,  't' },
					{"rsqrt",   no_argument,   0,  'n' },
					{"shift",   no_argument,   0,  'o' },
					{"rotate",  no_argument,   0,  'p' },
					{0,         0,             0,  0   }
			};
	while ((c = getopt_long_only(argc, argv, "agbcdefij", long_options, &option_index)) != -1){
		if (c == -1)
			break;

		switch (c)
		{
			case 'a':
				i8=1, i16=1, i32=1, i64=1;
				fp16=1, fp32=1, fp64=1;
				break;
			case 'g':
				i8 = 1;
				break;
			case 'b':
				i16 = 1;
				break;
			case 'c':
				i32 = 1;
				break;
			case 'd':
				i64 = 1;
				break;
			case 'v':
				fp16 = 1;
				break;
			case 'e':
				fp32 = 1;
				break;
			case 'f':
				fp64 = 1;
				break;
			case 'i':
				i8=1, i16=1, i32=1, i64=1;
				break;
			case 'j':
				fp16=1, fp32=1, fp64=1;
				break;
			case 'k':
				add = 1;
				break;
			case 'l':
				mul = 1;
				break;
			case 't':
				div = 1;
				break;
			case 'm':
				muladd = 1;
				break;
			case 'n':
				rsq = 1;
				break;
			case 'o':
				shift = 1;
				break;
			case 'p':
				rotate = 1;
				break;
			case 'h':
			default:
				printf("Usage: ./benchWarmer <args>\n");
				printf("\nArguments:\n");
				printf(" -h, show this help message and exit\n");
				printf("\n -a, run all tests (all datatypes, all operations)\n");
				
				printf("\n Datatype arguments (defaults to all):");
				printf("\n  --int, run all integer tests");
				printf("\n  --fp, run all FP tests\n");

				printf("\n  --int8, run int8 tests");
				printf("\n  --int16, run int16 tests");
				printf("\n  --int32, run int32 tests");
				printf("\n  --int64, run int64 tests\n");
				printf("\n  --fp16, run fp16 tests");
				printf("\n  --fp32, run fp32 tests");
				printf("\n  --fp64, run fp64 tests\n");

				printf("\n Operation arguments (defaults to all):");
				printf("\n  --add, run Add tests only");
				printf("\n  --mul, run Multiply tests only");
				printf("\n  --muladd, run MulAdd tests only");
				printf("\n  --div, run Division tests only");
				printf("\n  --rsqrt, run rsqrt tests only");
				printf("\n  --shift, run Shift tests only");
				printf("\n  --rotate, run Rotate tests only");
				printf("\n\n");
				exit(1);
		}
	}
	// If no operation is selected, do all of them
	if (! (add || mul || muladd || div || rsq || shift || rotate)) {
		add=1, mul=1, muladd=1, div=1, rsq=1, shift=1, rotate=1;
	}
	if (! (i8 || i16 || i32 || i64 || fp16 || fp32 || fp64)) {
		i8=1, i16=1, i32=1, i64=1, fp16=1, fp32=1, fp64=1;
	}
  printf("Datatype, Operation, Throughput mean (GFlops/s), Throughput stdev (GFlops/s), Duration mean (ms), Total flops, Total bytes accessed, AI, Workgroups, Threads, Experiments\n");
  if (i8) {
    bench_int<uint8_t>(add, mul, muladd, div, rsq, shift, rotate);
  }
  if (i16) {
		bench_int<uint16_t>(add, mul, muladd, div, rsq, shift, rotate);
  }
  if (i32) {
		bench_int<uint32_t>(add, mul, muladd, div, rsq, shift, rotate);
  }
  if (i64) {
		bench_int<uint64_t>(add, mul, muladd, div, rsq, shift, rotate);
  }
  if (fp16) {
		bench_fp<__half>(add, mul, muladd, div, rsq);
  }
  if (fp32) {
		bench_fp<float>(add, mul, muladd, div, rsq);
  }
  if (fp64) {
		bench_fp<double>(add, mul, muladd, div, rsq);
  }
}
