
#define LAMBDA [=]

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
  
    inline void syncDeviceData(void){}
  
    inline void syncHostData(void){}
  
    Udata(T *_ptr, uint _bytes) : ptr(_ptr), bytes(_bytes) {}

    Udata(const Udata& u) : ptr(u.ptr), d_ptr(u.d_ptr), bytes(u.bytes), is_copy(1) {}
  
    ~Udata(void){}
  
    inline T &operator [] (uint i) const {
      return ptr[i];
    }
  };

  inline static void init() {}

  template <typename Lambda>
  inline static void parallel_for(uint loop_size, Lambda loop_body) {
    for(uint i = 0; i < loop_size; i++){
      loop_body(i);
    }
  }
}
