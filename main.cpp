#include <stdio.h>

/* HAVE_DEF is set during compile time
 * and determines which accelerator backend is used
 * by including the respective header file */
#if defined(HAVE_CUDA)
  #include "devices_cuda.h"
#else
  #include "devices_host.h"
#endif

int main(int argc, char *argv []){

  /* Device initialization, currently
   * only required for GPU devices */
  devices::init();

  /* Declare pointers and allocate host memory */
  int *h_array, count = 10;
  h_array = (int*) malloc(count * sizeof(int));

  /* Create instance of unified data class that
   * emulates unified memory behavior without 
   * requiring support for unified memory */
  devices::Udata<int> array(h_array, count * sizeof(int)); 

  /* Initialize data from the host using
   * the standard syntax (ptr[index] = x) */
  for(uint i = 0; i < count; ++i){
    array[i] = i;
  }

  /* Sync data to device, this call
   * does nothing if no devices are present */
  array.syncDeviceData();

  /* Run a parallel for loop on the 
   * host or on a device depending on
   * the chosen compile configuration */
  devices::parallel_for(count, 
    LAMBDA(const uint i) {
      array[i] *= 2;
    }
  );

  /* Run a different parallel for loop on the 
   * host or on a device depending on
   * the chosen compile configuration */
  devices::parallel_for(count, 
    LAMBDA(const uint i) {
      if(array[i] == i * 2)
        printf("array[%u] = %d (which is correct!)\n", i, array[i]);
      else{
        printf("array[%u] = %d (which is incorrect!)\n", i, array[i]);
      }
    }
  );

  return 0;
}