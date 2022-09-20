
#define LAMBDA [=]

namespace devices
{
  inline static void init() {}

  inline static void* allocate(size_t bytes) {
    return malloc(bytes);
  }

  inline static void free(void* ptr) {
    ::free(ptr);
  }

  template <typename Lambda>
  inline static void parallel_for(uint loop_size, Lambda loop_body) {
    for(uint i = 0; i < loop_size; i++){
      loop_body(i);
    }
  }
}
