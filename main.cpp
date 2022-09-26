#include <chrono>
#include <iostream>
#include <stdio.h>

/* HAVE_DEF is set during compile time
 * and determines which accelerator backend is used
 * by including the respective header file */
#if defined(HAVE_CUDA)
  #include "devices_cuda.h"
#elif defined(HAVE_HIP)
  #include "devices_hip.h"
#else
  #include "devices_host.h"
#endif

int main(int argc, char *argv []){

  /* Device initialization, currently
   * only required for GPU devices */
  devices::init();

  /* Declare pointer and allocate host memory */
  int *array;

  /* Set the problem size and the number of dummy increments */
  uint n_array = 1e1, n_inc = 1e4;

  /* Begin timer */
  auto begin = std::chrono::steady_clock::now();
  
 /* Run a dummy increment loop to demonstrate
  * the performnace impact caused by recurring 
  * allocation and deallocation */
 for(uint inc = 0; inc < n_inc; ++inc){

    /* Allocate Unified Memory,
     * the standard malloc function is called
     * if running on host only */
    array = (int*) devices::allocate(n_array * sizeof(int));
  
    /* Initialize data from the host using
     * the standard syntax (ptr[index] = x) */
    for(uint i = 0; i < n_array; ++i){
      array[i] = i;
    }
  
    /* Run a parallel for loop on the 
     * host or on a device depending on
     * the chosen compile configuration */
    devices::parallel_for(n_array, 
      LAMBDA(const uint i) {
        array[i] *= 2;
      }
    );
  
    /* For the last increment, check the results
     * by running a different parallel for 
     * loop on the host or on a device depending 
     * on the chosen compile configuration */
    if(inc == n_inc - 1){
      devices::parallel_for(n_array, 
        LAMBDA(const uint i) {
          if(array[i] == i * 2)
            printf("array[%u] = %d (which is correct!)\n", i, array[i]);
          else{
            printf("array[%u] = %d (which is incorrect!)\n", i, array[i]);
          }
        }
      );
    }
  
    /* Free the Unified Memory allocation,
     * the standard free function is called
     * if running on host only */
    devices::free(array);
  }

  /* Print timing */
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>
    (std::chrono::steady_clock::now() - begin).count() << "[ms]" << std::endl;

  return 0;
}