amd: pubBench-amd
nv: pubBench-nv
nvidia: nv

# Set number of computes
numFMA ?= 10000

%-nv: %.cu
		nvcc $^ -o $@ -O3 -D numFMA=$(numFMA)
%-amd: %.cu
		hipcc -std=c++11 $^ -o $@ -O3 -D numFMA=$(numFMA)
		
clean: 
	rm pubBench-*
