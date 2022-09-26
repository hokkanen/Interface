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

  /* Set the problem size and the number of dummy increments */
  uint n_array = 1e1, n_inc = 1e4;
  
  /* Declare pointer and allocate host memory */
  int *host_array = (int*) malloc(n_array * sizeof(int));

  /* Begin timer */
  auto begin = std::chrono::steady_clock::now();
  
  /* Run a dummy increment loop to demonstrate
   * the performnace impact caused by recurring 
   * allocation and deallocation */
  for(uint inc = 0; inc < n_inc; ++inc){

    /* Create instance of unified data class that
     * emulates Unified Memory behavior without 
     * requiring support for Unified Memory */
    devices::Udata<int> array(host_array, n_array * sizeof(int)); 
  
    /* Initialize data from the host using
     * the standard syntax (ptr[index] = x) */
    for(uint i = 0; i < n_array; ++i){
      array[i] = i;
    }

    /* Sync data to device, this call
     * does nothing if no devices are present */
    array.syncDeviceData();
  
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
  }

  /* Print timing */
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>
    (std::chrono::steady_clock::now() - begin).count() << "[ms]" << std::endl;

  return 0;
}