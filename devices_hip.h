#include <hip/hip_runtime.h>

#if defined(HAVE_UMPIRE)
  #include "umpire/interface/c_fortran/umpire.h"
#endif

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
  __forceinline__ static void init() {
    int device_id = 0;
    HIP_ERR(hipSetDevice(device_id));

    #if defined(HAVE_UMPIRE)
      umpire_resourcemanager rm;
      umpire_resourcemanager_get_instance(&rm);
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(&rm, "UM", &allocator);
      umpire_allocator pool;
      umpire_resourcemanager_make_allocator_quick_pool(&rm, "pool", allocator, 1024, 1024, &pool);
    #endif
  }

  __forceinline__ static void* allocate(size_t bytes) {
    void* ptr;
    #if defined(HAVE_UMPIRE)
      umpire_resourcemanager rm;
      umpire_resourcemanager_get_instance(&rm);
      umpire_allocator pool;
      umpire_resourcemanager_get_allocator_by_name(&rm, "pool", &pool);
      ptr = umpire_allocator_allocate(&pool, bytes);
    #else
      HIP_ERR(hipMallocManaged(&ptr, bytes));
    #endif
    return ptr;
  }

  __forceinline__ static void free(void* ptr) {
    #if defined(HAVE_UMPIRE)
      umpire_resourcemanager rm;
      umpire_resourcemanager_get_instance(&rm);
      umpire_allocator pool;
      umpire_resourcemanager_get_allocator_by_name(&rm, "pool", &pool);
      umpire_allocator_deallocate(&pool, ptr);
    #else
      HIP_ERR(hipFree(ptr));
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
