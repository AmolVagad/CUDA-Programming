#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint32_t* bins);
/* Include below the function headers of any other functions that you implement */

void preallocate_memory(uint32_t* input,size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH], uint32_t * temp_input);

void deallocate_memory(uint32_t * host_bins, uint32_t * device_bin, size_t histo_height, size_t histo_width);


// Function to allocate devie memory 

 uint32_t* AllocateDevice(size_t size);

//  Function to copy to device
 void  HostToDevice(uint32_t* d_data, uint32_t* h_data, size_t size);

void setmemory(uint32_t* data, int value , size_t count);



// Function to copy from deice to host

void DeviceToHost(uint32_t* h_data, uint32_t* d_data, size_t size);

  //Function to free device memory 

void DeviceFree(uint32_t *d_data);
  


extern uint32_t *d_input;
extern uint32_t *device_bins;
#endif
