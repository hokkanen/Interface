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
  template <typename T> 
  class Udata {

    private:
    
    T *ptr; 
    T *d_ptr;
    uint bytes;
    uint is_copy = 0;
  
    public:   
  
    __forceinline__ void syncDeviceData(void){
      CUDA_ERR(cudaMemcpy(d_ptr, ptr, bytes, cudaMemcpyHostToDevice));
    }
  
    __forceinline__ void syncHostData(void){
      CUDA_ERR(cudaMemcpy(ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));
    }
  
    Udata(T *_ptr, uint _bytes) : ptr(_ptr), bytes(_bytes) {
      #if defined(CUDA_MEMPOOL)
        CUDA_ERR(cudaMallocAsync(&d_ptr, bytes, 0));
      #else
        CUDA_ERR(cudaMalloc(&d_ptr, bytes));
      #endif  
      CUDA_ERR(cudaMemcpy(d_ptr, ptr, bytes, cudaMemcpyHostToDevice));
    }
    
    __host__ __device__ Udata(const Udata& u) : 
      ptr(u.ptr), d_ptr(u.d_ptr), bytes(u.bytes), is_copy(1) {}
  
    __host__ __device__ ~Udata(void){
      if(!is_copy){
        #if defined(CUDA_MEMPOOL)
          cudaFreeAsync(d_ptr, 0);
        #else  
          cudaFree(d_ptr);
        #endif
      }
    }
  
    __forceinline__ __host__ __device__ T &operator [] (uint i) const {
     #ifdef __CUDA_ARCH__
        return d_ptr[i];
     #else
        return ptr[i];
     #endif
    }
  };
  
  __forceinline__ static void init() {
    int device_id = 0;
    CUDA_ERR(cudaSetDevice(device_id));
    #if defined(CUDA_MEMPOOL)
      cudaMemPool_t mempool;
      CUDA_ERR(cudaDeviceGetDefaultMemPool(&mempool, device_id));
      uint64_t threshold = UINT64_MAX;
      CUDA_ERR(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    #endif
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
