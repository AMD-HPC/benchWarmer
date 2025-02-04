amd: benchWarmer-amd
nv: benchWarmer-nv
nvidia: nv

# Set number of computes
nOps ?= 1000

%-nv: %.cu
		nvcc $^ -o $@ -O3 -D nOps=$(nOps)
%-amd: %.cu
		hipcc -std=c++11 $^ -o $@ -O3 -D nOps=$(nOps)
		
clean: 
	rm benchWarmer-*
