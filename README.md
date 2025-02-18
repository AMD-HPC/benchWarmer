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
  --xor, run XOR tests only
  --shift, run Shift tests only
  --rotate, run Rotate tests only
```

### Sample Run
`./benchWarmer-amd --add --mul --muladd --div --rsqrt`
```
Running int8 tests:
  Add test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 1073741824, experiments: 10
    Total FLOPS=1073741824000, total bytes accessed=2684354560, AI=400.000000, mean duration=103.380 ms

    Mean throughput=10386.333984 GFLOPs/sec, stdev=3.606 GFLOPs/s, 95% Confidence Interval: [10384.099, 10388.569]

  Mul test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 1073741824, experiments: 10
    Total FLOPS=1073741824000, total bytes accessed=2684354560, AI=400.000000, mean duration=103.492 ms

    Mean throughput=10375.151367 GFLOPs/sec, stdev=2.299 GFLOPs/s, 95% Confidence Interval: [10373.727, 10376.576]

  MulAdd test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 1073741824, experiments: 10
    Total FLOPS=2147483648000, total bytes accessed=2684354560, AI=800.000000, mean duration=159.284 ms

    Mean throughput=13482.132812 GFLOPs/sec, stdev=2.213 GFLOPs/s, 95% Confidence Interval: [13480.762, 13483.504]

  Div test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 1073741824, experiments: 10
    Total FLOPS=1073741824000, total bytes accessed=2684354560, AI=400.000000, mean duration=931.007 ms

    Mean throughput=1153.312744 GFLOPs/sec, stdev=0.074 GFLOPs/s, 95% Confidence Interval: [1153.267, 1153.358]

  Rsqrt test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 1073741824, experiments: 10
    Total FLOPS=1073741824000, total bytes accessed=2684354560, AI=400.000000, mean duration=463.321 ms

    Mean throughput=2317.488770 GFLOPs/sec, stdev=0.164 GFLOPs/s, 95% Confidence Interval: [2317.387, 2317.590]


Running int16 tests:
  Add test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 536870912, experiments: 10
    Total FLOPS=536870912000, total bytes accessed=2684354560, AI=200.000000, mean duration=51.826 ms

    Mean throughput=10359.085938 GFLOPs/sec, stdev=4.751 GFLOPs/s, 95% Confidence Interval: [10356.141, 10362.031]

  Mul test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 536870912, experiments: 10
    Total FLOPS=536870912000, total bytes accessed=2684354560, AI=200.000000, mean duration=51.838 ms

    Mean throughput=10356.744141 GFLOPs/sec, stdev=1.737 GFLOPs/s, 95% Confidence Interval: [10355.668, 10357.820]

  MulAdd test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 536870912, experiments: 10
    Total FLOPS=1073741824000, total bytes accessed=2684354560, AI=400.000000, mean duration=79.728 ms

    Mean throughput=13467.551758 GFLOPs/sec, stdev=3.109 GFLOPs/s, 95% Confidence Interval: [13465.625, 13469.479]

  Div test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 536870912, experiments: 10
    Total FLOPS=536870912000, total bytes accessed=2684354560, AI=200.000000, mean duration=466.174 ms

    Mean throughput=1151.653076 GFLOPs/sec, stdev=0.126 GFLOPs/s, 95% Confidence Interval: [1151.575, 1151.731]

  Rsqrt test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 536870912, experiments: 10
    Total FLOPS=536870912000, total bytes accessed=2684354560, AI=200.000000, mean duration=231.746 ms

    Mean throughput=2316.630371 GFLOPs/sec, stdev=0.241 GFLOPs/s, 95% Confidence Interval: [2316.481, 2316.780]


Running int32 tests:
  Add test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=26.217 ms

    Mean throughput=10239.052734 GFLOPs/sec, stdev=12.942 GFLOPs/s, 95% Confidence Interval: [10231.031, 10247.074]

  Mul test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=26.087 ms

    Mean throughput=10289.828125 GFLOPs/sec, stdev=1.773 GFLOPs/s, 95% Confidence Interval: [10288.729, 10290.927]

  MulAdd test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=536870912000, total bytes accessed=2684354560, AI=200.000000, mean duration=41.693 ms

    Mean throughput=12876.739258 GFLOPs/sec, stdev=2.003 GFLOPs/s, 95% Confidence Interval: [12875.497, 12877.981]

  Div test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=431.025 ms

    Mean throughput=622.784363 GFLOPs/sec, stdev=0.049 GFLOPs/s, 95% Confidence Interval: [622.754, 622.815]

  Rsqrt test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=115.656 ms

    Mean throughput=2320.987793 GFLOPs/sec, stdev=0.344 GFLOPs/s, 95% Confidence Interval: [2320.774, 2321.201]


Running int64 tests:
  Add test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=134217728000, total bytes accessed=2684354560, AI=50.000000, mean duration=26.775 ms

    Mean throughput=5012.807617 GFLOPs/sec, stdev=2.666 GFLOPs/s, 95% Confidence Interval: [5011.155, 5014.460]

  Mul test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=134217728000, total bytes accessed=2684354560, AI=50.000000, mean duration=52.436 ms

    Mean throughput=2559.640137 GFLOPs/sec, stdev=0.329 GFLOPs/s, 95% Confidence Interval: [2559.436, 2559.844]

  MulAdd test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=54.724 ms

    Mean throughput=4905.236816 GFLOPs/sec, stdev=3.822 GFLOPs/s, 95% Confidence Interval: [4902.868, 4907.605]

  Div test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=134217728000, total bytes accessed=2684354560, AI=50.000000, mean duration=331.087 ms

    Mean throughput=405.385559 GFLOPs/sec, stdev=0.143 GFLOPs/s, 95% Confidence Interval: [405.297, 405.474]

  Rsqrt test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=134217728000, total bytes accessed=2684354560, AI=50.000000, mean duration=136.660 ms

    Mean throughput=982.132324 GFLOPs/sec, stdev=0.159 GFLOPs/s, 95% Confidence Interval: [982.034, 982.231]


Running FP32 tests:
  Add test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=26.070 ms

    Mean throughput=10296.844727 GFLOPs/sec, stdev=2.630 GFLOPs/s, 95% Confidence Interval: [10295.215, 10298.475]

  Mul test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=25.926 ms

    Mean throughput=10354.003906 GFLOPs/sec, stdev=2.394 GFLOPs/s, 95% Confidence Interval: [10352.521, 10355.487]

  MulAdd test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=536870912000, total bytes accessed=2684354560, AI=200.000000, mean duration=26.849 ms

    Mean throughput=19996.115234 GFLOPs/sec, stdev=2.328 GFLOPs/s, 95% Confidence Interval: [19994.672, 19997.559]

  Div test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=367.487 ms

    Mean throughput=730.461975 GFLOPs/sec, stdev=0.104 GFLOPs/s, 95% Confidence Interval: [730.398, 730.526]

  Rsqrt test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 268435456, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=115.116 ms

    Mean throughput=2331.865723 GFLOPs/sec, stdev=0.194 GFLOPs/s, 95% Confidence Interval: [2331.745, 2331.986]


Running FP64 tests:
  Add test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=134217728000, total bytes accessed=2684354560, AI=50.000000, mean duration=13.553 ms

    Mean throughput=9902.954102 GFLOPs/sec, stdev=1.844 GFLOPs/s, 95% Confidence Interval: [9901.811, 9904.098]

  Mul test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=134217728000, total bytes accessed=2684354560, AI=50.000000, mean duration=13.536 ms

    Mean throughput=9915.560547 GFLOPs/sec, stdev=0.836 GFLOPs/s, 95% Confidence Interval: [9915.042, 9916.079]

  MulAdd test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=268435456000, total bytes accessed=2684354560, AI=100.000000, mean duration=13.644 ms

    Mean throughput=19673.941406 GFLOPs/sec, stdev=3.045 GFLOPs/s, 95% Confidence Interval: [19672.055, 19675.828]

  Div test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=134217728000, total bytes accessed=2684354560, AI=50.000000, mean duration=190.419 ms

    Mean throughput=704.856567 GFLOPs/sec, stdev=0.128 GFLOPs/s, 95% Confidence Interval: [704.777, 704.936]

  Rsqrt test: workgroupSize:256, workgroups:16384, nThreads: 4194304, nSize: 134217728, experiments: 10
    Total FLOPS=134217728000, total bytes accessed=2684354560, AI=50.000000, mean duration=57.621 ms

    Mean throughput=2329.316895 GFLOPs/sec, stdev=0.408 GFLOPs/s, 95% Confidence Interval: [2329.064, 2329.570]
```
