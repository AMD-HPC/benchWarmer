# benchWarmer
benchWarmer is a synthetic microbenchmark suite that measures throughput of various operations on integer and floating-point datatypes. This tool is compatible with both AMD and NVIDIA architectures.

## Build
For AMD architectures:

`make amd nOps=<# of ops>`

For NVIDIA architectures:

`make nv nOps=<# of ops>`

The optional `nOps` argument (1000 by default) controls how many operations that will be computed on each piece of data. By modifying this number, one can adjust the Arithmetic Intensity of the workload.

## Run
`./benchWarmer-amd <args>`
or
`./benchWarmer-nv <args>`

Output will be in a CSV format and sent to the console. One can redirect the output to a file using `>`.

```
Arguments:
 -h, show this help message and exit

 -a, run all tests (all datatypes, all operations)

 Datatype arguments (defaults to all):
  --int, run all integer tests
  --fp, run all FP tests

  --int8, run int8 tests
  --int16, run int16 tests
  --int32, run int32 tests
  --int64, run int64 tests

  --fp16, run fp16 tests
  --fp32, run fp32 tests
  --fp64, run fp64 tests

 Operation arguments (defaults to all):
  --add, run Add tests only
  --mul, run Multiply tests only
  --muladd, run MulAdd tests only
  --div, run Division tests only
  --rsqrt, run rsqrt tests only
  --shift, run Shift tests only
  --rotate, run Rotate tests only
  --convert, run Type Conversion tests only
```

