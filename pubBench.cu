#if __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>

#define gpu(symbol) cuda ## symbol

#elif __HIPCC__

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#define gpu(symbol) hip ## symbol

#endif

#include <string>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Number of computes must be set at compile time
#ifndef nOps
#define nOps 10000
#endif

#define DEFAULT_WORKGROUP_SIZE 256
#define DEFAULT_WORKGROUPS 16384
#define DEFAULT_NUM_EXPERIMENTS 10
#define DEFAULT_DATASET_SIZE 1024 * 1024 * 1024

// function classes that we want to measure
// TODO update comments

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
struct FMA {
  __device__ T operator()(T x, T y, T z) {
    return x * y + z;
  }
};

template<class T>
struct Rsqrt {
  __device__ T operator()(T x, T y, T z) {
    return rsqrtf(x);
  }
};

template<class T>
struct Xor {
  __device__ T operator()(T x, T y, T z) {
    return x ^ y;
  }
};

template<class T>
struct Choosery {
  __device__ T operator()(T x, T y, T z) {
    return x ^ ((x ^ y) & z); // (x & ~z) | (y & z)
  }
};

template<class T>
struct Majority1 {
  __device__ T operator()(T x, T y, T z) {
    Choosery<T> ch;
    return ch(x, y, (x ^ z));
  }
};

template<class T>
struct Majority2 {
  __device__ T operator()(T x, T y, T z) {
    return (x & y) ^ ((x ^ y) & z);
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

template<class T, int N>
struct ShiftLeftImm {
  __device__ T operator()(T x, T y, T z) {
    return x << N;
  }
};

template<class T, int N>
struct ShiftRightImm {
  __device__ T operator()(T x, T y, T z) {
    return x >> N;
  }
};

template<class T, int N>
struct RotateLeftImm {
  __device__ T operator()(T x, T y, T z) {
    return (x << N) | (x >> (8 * sizeof(T) - N));
  }
};

template<class T, int N>
struct RotateRightImm {
  __device__ T operator()(T x, T y, T z) {
    return (x >> N) | (x << (8 * sizeof(T) - N));
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
		for(int j=0; j<n; j+=2)
		{
			// Two different write locations to force the compiler to complete every operation
			x = func(a[offset], x, y);
			a[offset] = func(a[offset], x, y);
		}
	}
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
	std::string s = typeid(Func).name();
	if(s.find("FMA") != std::string::npos) {
		totalFlops *= 2;
	}
  uint64_t totalBytes = (uint64_t)nSize * (uint64_t)sizeof(T) * 2.5;

  assert((gpu(Malloc(&memBlock, DEFAULT_DATASET_SIZE)))==gpu(Success));

  throughput_kernel<T,nOps,Func><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((T *)memBlock, nSize);
  gpu(DeviceSynchronize());

	// Timing data
  float eventMs;
  gpu(Event_t) start, stop;

  // Measurement data 
  float meanThroughput, stdevThroughput, confidenceThroughput;
  float meanDuration, stdevDuration, confidenceDuration;
  float *throughputs = (float *)calloc(numExperiments, sizeof(float));
  float *durations = (float *)calloc(numExperiments, sizeof(float));

	// Run experiments
  for (int n=0; n<numExperiments; n++)
  {
    initTimeEvents(start, stop);
		throughput_kernel<T,nOps,Func><<<dim3(numWorkgroups), dim3(workgroupSize)>>>((T *)memBlock, nSize);
    stopTimeEvents(eventMs, start, stop);

    throughputs[n] = (float) totalFlops / eventMs / 1e6;  // Unit: GFLOPs/sec
    durations[n] = eventMs;
  }

  // Calculate summary statistics
  stats(throughputs, numExperiments, &meanThroughput, &stdevThroughput, &confidenceThroughput);
  stats(durations, numExperiments, &meanDuration, &stdevDuration, &confidenceDuration);

	// Print output
  printf("workgroupSize:%d, workgroups:%d, nThreads: %lu, nSize: %d, experiments: %d\n",
      workgroupSize, numWorkgroups, nThreads, nSize, numExperiments);
  printf("    Total FLOPS=%lu, total bytes accessed=%lu, AI=%f, mean duration=%.3f ms\n\n",
      totalFlops, totalBytes, ((float)totalFlops/(float)totalBytes), meanDuration);
  printf("    Mean throughput=%f GFLOPs/sec, stdev=%.3f GFLOPs/s, 95%% Confidence Interval: [%.3f, %.3f]\n\n",
      meanThroughput, stdevThroughput, meanThroughput - confidenceThroughput, meanThroughput + confidenceThroughput);

  // Clean up time
  gpu(Free(memBlock));
}

template<class T>
static void bench_int(bool add, bool mul, bool fma, bool rsq, bool xorFunc, bool shift, bool rotate, bool choosery, bool majority) {
	if(add) {
		printf("  Add test: ");
		bench_func<T,Add<T>>();
	}
	if(mul) {
		printf("  Mul test: ");
		bench_func<T,Mul<T>>();
	}
	if(fma) {
		printf("  FMA test: ");
		bench_func<T,FMA<T>>();
	}
	if(rsq) {
		printf("  Rsqrt test: ");
		bench_func<T,FMA<T>>();
	}
	if(xorFunc) {
		printf("  Xor test: ");
		bench_func<T,Xor<T>>();
	}
	if(shift) {
		printf("  ShiftLeft test: ");
		bench_func<T,ShiftLeft<T>>();
		printf("  ShiftRight test: ");
		bench_func<T,ShiftRight<T>>();
		printf("  ShiftLeftImm test: ");
		bench_func<T,ShiftLeftImm<T,3>>();
		printf("  ShiftRightImm test: ");
		bench_func<T,ShiftRightImm<T,3>>();
	}
	if(rotate) {
		printf("  RotateLeft test: ");
		bench_func<T,RotateLeft<T>>();
		printf("  RotateRight test: ");
		bench_func<T,RotateRight<T>>();
		printf("  RotateLeftImm test: ");
		bench_func<T,RotateLeftImm<T,3>>();
		printf("  RotateRightImm test: ");
		bench_func<T,RotateRightImm<T,3>>();
	}
	if(choosery) {
		printf("  Choosery test: ");
		bench_func<T,Choosery<T>>();
	}
	if(majority) {
		printf("  Majority1 test: ");
		bench_func<T,Majority1<T>>();
		printf("  Majority2 test: ");
		bench_func<T,Majority2<T>>();
	}
}


template<class T>
static void bench_float(bool add, bool mul, bool fma, bool rsq) {
	if(add) {
		printf("  Add test: ");
		bench_func<T,Add<T>>();
	}
	if(mul) {
		printf("  Mul test: ");
		bench_func<T,Mul<T>>();
	}
	if(fma) {
		printf("  FMA test: ");
		bench_func<T,FMA<T>>();
	}
	if(rsq) {
		printf("  Rsqrt test: ");
		bench_func<T,Rsqrt<T>>();
	}
}

int main(int argc, char **argv)
{

	//CLI parsing
	bool i8 = false, i16 = false, i32 = false, i64 = false, fp32 = false, fp64 = false;
	bool add = false, mul = false, fma = false, rsq = false, xorFunc = false, shift = false, rotate = false, choosery = false, majority = false;

	int c, option_index = 0;
	static struct option long_options[] = {
					{"h",       no_argument,   0,  'h' },
					{"a",       no_argument,   0,  'a' },
					{"int8",    no_argument,   0,  'g' },
					{"int16",   no_argument,   0,  'b' },
					{"int32",   no_argument,   0,  'c' },
					{"int64",   no_argument,   0,  'd' },
					{"fp32",    no_argument,   0,  'e' },
					{"fp64",    no_argument,   0,  'f' },
					{"int",     no_argument,   0,  'i' },
					{"fp",      no_argument,   0,  'j' },
					{"add",     no_argument,   0,  'k' },
					{"mul",     no_argument,   0,  'l' },
					{"fma",     no_argument,   0,  'm' },
					{"rsqrt",   no_argument,   0,  'n' },
					{"xor",     no_argument,   0,  's' },
					{"shift",   no_argument,   0,  'o' },
					{"rotate",  no_argument,   0,  'p' },
					{"choosery",no_argument,   0,  'q' },
					{"majority",no_argument,   0,  'r' },
					{0,         0,             0,  0   }
			};
	while ((c = getopt_long_only(argc, argv, "agbcdefij", long_options, &option_index)) != -1){
		if (c == -1)
			break;

		switch (c)
		{
			case 'a':
				i8=1, i16=1, i32=1, i64=1;
				fp32=1, fp64=1;
				printf("Selected all tests\n");
				break;
			case 'g':
				i8 = 1;
				printf("Selected int8 test\n");
				break;
			case 'b':
				i16 = 1;
				printf("Selected int16 test\n");
				break;
			case 'c':
				i32 = 1;
				printf("Selected int32 test\n");
				break;
			case 'd':
				i64 = 1;
				printf("Selected int64 test\n");
				break;
			case 'e':
				fp32 = 1;
				printf("Selected FP32 test\n");
				break;
			case 'f':
				fp64 = 1;
				printf("Selected FP64 test\n");
				break;
			case 'i':
				i8=1, i16=1, i32=1, i64=1;
				printf("Selected all int tests\n");
				break;
			case 'j':
				fp32=1, fp64=1;
				printf("Selected all FP tests\n");
				break;
			case 'k':
				add = 1;
				printf("Selected Add test\n");
				break;
			case 'l':
				mul = 1;
				printf("Selected Multiply test\n");
				break;
			case 'm':
				fma = 1;
				printf("Selected FMA test\n");
				break;
			case 'n':
				rsq = 1;
				printf("Selected Rsqrt test\n");
				break;
			case 's':
				xorFunc = 1;
				printf("Selected XOR test\n");
				break;
			case 'o':
				shift = 1;
				printf("Selected Shift test\n");
				break;
			case 'p':
				rotate = 1;
				printf("Selected Rotate test\n");
				break;
			case 'q':
				choosery = 1;
				printf("Selected Choosery test\n");
				break;
			case 'r':
				majority = 1;
				printf("Selected Majority test\n");
				break;
			case 'h':
			default:
				printf("Usage: ./pubBench <args>\n");
				printf("\nArguments:\n");
				printf(" -h, show this help message and exit\n");
				printf("\n -a, run all tests (all datatypes, all operations)\n");
				
				printf("\n Datatype arguments (defaults to all):");
				printf("\n  --int, run all integer tests");
				printf("\n  --fp, run all FP tests\n");

				printf("\n  --int8, run int8 tests");
				printf("\n  --int16, run int16 tests");
				printf("\n  --int32, run int32 tests");
				printf("\n  --int64, run int64 tests");
				printf("\n  --fp32, run fp32 tests");
				printf("\n  --fp64, run fp64 tests\n");

				printf("\n Operation arguments (defaults to all):");
				printf("\n  --add, run Add tests only");
				printf("\n  --mul, run Multiply tests only");
				printf("\n  --fma, run FMA tests only");
				printf("\n  --rsqrt, run rsqrt tests only");
				printf("\n  --xor, run XOR tests only");
				printf("\n  --shift, run Shift tests only");
				printf("\n  --rotate, run Rotate tests only");
				printf("\n  --choosery, run Choosery tests only");
				printf("\n  --majority, run Majority tests only");
				printf("\n\n");
				exit(1);
		}
	}
	// If no operation is selected, do all of them
	if (! (add || mul || fma || rsq || xorFunc || shift || rotate || choosery || majority)) {
		printf("Selected all operation tests\n");
		add=1, mul=1, fma=1, rsq=1, xorFunc=1, shift=1, rotate=1, choosery=1, majority=1;
	}
	if (! (i8 || i16 || i32 || i64 || fp32 || fp64)) {
		printf("Selected all datatype tests\n");
		i8=1, i16=1, i32=1, i64=1, fp32=1, fp64=1;
	}
  if (i8) {
    printf("\nRunning int8 tests:\n");
    bench_int<uint8_t>(add, mul, fma, rsq, xorFunc, shift, rotate, choosery, majority);
  }
  if (i16) {
		printf("\nRunning int16 tests:\n");
		bench_int<uint16_t>(add, mul, fma, rsq, xorFunc, shift, rotate, choosery, majority);
  }
  if (i32) {
		printf("\nRunning int32 tests:\n");
		bench_int<uint32_t>(add, mul, fma, rsq, xorFunc, shift, rotate, choosery, majority);
  }
  if (i64) {
		printf("\nRunning int64 tests:\n");
		bench_int<uint64_t>(add, mul, fma, rsq, xorFunc, shift, rotate, choosery, majority);
  }
  if (fp32) {
		printf("\nRunning FP32 tests:\n");
		bench_float<float>(add, mul, fma, rsq);
  }
  if (fp64) {
		printf("\nRunning FP64 tests:\n");
		bench_float<double>(add, mul, fma, rsq);
  }
}
