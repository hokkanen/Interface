#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_ERR(err) (cuda_error(err, __FILE__, __LINE__))
inline static void cuda_error(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

#define LAMBDA [=] __host__ __device__

namespace devices
{
  __forceinline__ static void init() {
    int device_id = 0;
    CUDA_ERR(cudaSetDevice(device_id));
  }

  __forceinline__ static void* allocate(size_t bytes) {
    void* ptr;
    CUDA_ERR(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  __forceinline__ static void free(void* ptr) {
    CUDA_ERR(cudaFree(ptr));
  }
  
  template <typename LambdaBody> 
  __global__ static void cudaKernel(LambdaBody lambda, const uint loop_size)
  {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < loop_size)
    {
      lambda(i);
    }
  }
  
  template <typename T>
  __forceinline__ static void parallel_for(uint loop_size, T loop_body) {
    const uint blocksize = 64;
    const uint gridsize = (loop_size - 1 + blocksize) / blocksize;
    cudaKernel<<<gridsize, blocksize>>>(loop_body, loop_size);
    CUDA_ERR(cudaStreamSynchronize(0));
  }
}
