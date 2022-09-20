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
  int *array, count = 10;
  
  /* Allocate Unified Memory,
   * the standard malloc function is called
   * if running on host only */
  array = (int*) devices::allocate(count * sizeof(int));

  /* Initialize data from the host using
   * the standard syntax (ptr[index] = x) */
  for(uint i = 0; i < count; ++i){
    array[i] = i;
  }

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

  /* Free the Unified Memory allocation,
   * the standard free function is called
   * if running on host only */
  devices::free(array);

  return 0;
}