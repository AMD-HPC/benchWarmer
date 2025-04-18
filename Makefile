ROCSTAR_ROOT ?= "../rocSTAR/build"
EXP ?= 20000
amd: benchWarmer-amd_$(GPU)_$(nOps)_$(EXP)
nv: benchWarmer-nv_$(GPU)_$(nOps)_$(EXP)
nvidia: nv

# Set number of computes
nOps ?= 1000

%-nv_$(GPU)_$(nOps)_$(EXP): %.cu
		nvcc $^ -o $@ -O3 -D nOps=$(nOps)
%-amd_$(GPU)_$(nOps)_$(EXP): %.cu
		hipcc -std=c++11 $^ -o $@ -O3 -D nOps=$(nOps) -D EXP=$(EXP) -I "$(ROCSTAR_ROOT)/include" -L "$(ROCSTAR_ROOT)/lib" -lrocSTAR
		
clean: 
	rm benchWarmer-*
