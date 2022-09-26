#include <hip/hip_runtime.h>

#define HIP_ERR(err) (hip_error(err, __FILE__, __LINE__))
inline static void hip_error(hipError_t err, const char *file, int line) {
	if (err != hipSuccess) {
		printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
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
      HIP_ERR(hipMemcpy(d_ptr, ptr, bytes, hipMemcpyHostToDevice));
    }
  
    __forceinline__ void syncHostData(void){
      HIP_ERR(hipMemcpy(ptr, d_ptr, bytes, hipMemcpyDeviceToHost));
    }
  
    Udata(T *_ptr, uint _bytes) : ptr(_ptr), bytes(_bytes) {
      #if defined(HIP_MEMPOOL)
        HIP_ERR(hipMallocAsync(&d_ptr, bytes, 0));
      #else
        HIP_ERR(hipMalloc(&d_ptr, bytes));
      #endif  
      HIP_ERR(hipMemcpy(d_ptr, ptr, bytes, hipMemcpyHostToDevice));
    }
    
    __host__ __device__ Udata(const Udata& u) : 
      ptr(u.ptr), d_ptr(u.d_ptr), bytes(u.bytes), is_copy(1) {}
  
    __host__ __device__ ~Udata(void){
      if(!is_copy){
        #if defined(HIP_MEMPOOL)
          hipFreeAsync(d_ptr, 0);
        #else  
          hipFree(d_ptr);
        #endif
      }
    }
  
    __forceinline__ __host__ __device__ T &operator [] (uint i) const {
     #ifdef __HIP_ARCH__
        return d_ptr[i];
     #else
        return ptr[i];
     #endif
    }
  };
  
  __forceinline__ static void init() {
    int device_id = 0;
    HIP_ERR(hipSetDevice(device_id));
    #if defined(HIP_MEMPOOL)
      hipMemPool_t mempool;
      HIP_ERR(hipDeviceGetDefaultMemPool(&mempool, device_id));
      uint64_t threshold = UINT64_MAX;
      HIP_ERR(hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold, &threshold));
    #endif
  }
  
  template <typename LambdaBody> 
  __global__ static void hipKernel(LambdaBody lambda, const uint loop_size)
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
    hipKernel<<<gridsize, blocksize>>>(loop_body, loop_size);
    HIP_ERR(hipStreamSynchronize(0));
  }
}
