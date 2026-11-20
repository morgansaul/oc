NVCC ?= nvcc
CXX ?= g++
CUDA_ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -lineinfo
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

.PHONY: all clean test-core test-openssl
all: xonly_cuda

xonly_cuda: xonly_cuda.cu secp_core.cuh
	$(NVCC) $(NVCCFLAGS) xonly_cuda.cu -o xonly_cuda

core_test: core_test.cpp secp_core.cuh
	$(CXX) $(CXXFLAGS) core_test.cpp -o core_test

test-core: core_test
	./core_test

openssl_check: openssl_check.cpp secp_core.cuh
	$(CXX) $(CXXFLAGS) openssl_check.cpp -lcrypto -o openssl_check

test-openssl: openssl_check
	./openssl_check

clean:
	rm -f xonly_cuda core_test core_test_new openssl_check
