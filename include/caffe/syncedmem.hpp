/**
 *  增加caffe对OpenCL的支持，这个支持跟是否又GPU无关
 *  因为很多CPU也同样支持OpenCL. CUDA和OpenCL不能同时使用. 
 *  无论在编译的时候是否使用了CPU_ONLY依然表示支持OpenCL.
 *  目前的版本仅支持单GPU
 *  修改作者：尚庆东,邮箱：shang_qd@qq.com
 */

#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

	// If CUDA is available and in GPU mode, host memory will be allocated pinned,
	// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
	// The improvement in performance seems negligible in the single GPU case,
	// but might be more significant for parallel training. Most importantly,
	// it improved stability for large models on many GPUs.
	// 如果是OpenCL模式,使用free,malloc 方式申请内存，且cpu_malloc_use_cuda_永远都是false
	inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda)
	{
		if (Caffe::mode() == Caffe::CL)
		{
			*ptr = malloc(size);
			*use_cuda = false;
			CHECK(*ptr) << "host allocation of size " << size << " failed";
		}
		else
		{
#ifndef CPU_ONLY
			if (Caffe::mode() == Caffe::GPU) {
				CUDA_CHECK(cudaMallocHost(ptr, size));
				*use_cuda = true;
				return;
			}
#endif
			*ptr = malloc(size);
			*use_cuda = false;
			CHECK(*ptr) << "host allocation of size " << size << " failed";
		}
	}

	// 如果是OpenCL模式,使用free,malloc 方式申请内存，且cpu_malloc_use_cuda_永远都是false
	inline void CaffeFreeHost(void* ptr, bool use_cuda)
	{
		if (Caffe::mode() == Caffe::CL)
		{
			free(ptr);
		}
		else
		{
#ifndef CPU_ONLY
			if (use_cuda) {
				CUDA_CHECK(cudaFreeHost(ptr));
				return;
			}
#endif
			free(ptr);
		}
	}

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  
  void* mutable_cpu_data();
  void* mutable_gpu_data();

  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  
  // 
  SyncedHead head() { return head_; }

  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();
  void to_gpu();

  void* cpu_ptr_;
  // 因为cuda和OpenCL不能同时支持，GPU模式下gpu_ptr_ 表示cuda 指针
  // 在CL模式下，表示openCL 指针，下面的变量同理
  void* gpu_ptr_;

  size_t size_;
  SyncedHead head_;
  // cpu_data_ 数据是否在本类中申请的，在本类中为true,是外部就是false
  bool own_cpu_data_;
  // 如果是使用上面的两个函数来处理内存，那就是内存支持CPU和GPU内存共享
  bool cpu_malloc_use_cuda_;
  // gpu_data_ 数据是否在本类中申请的，在本类中为true,是外部就是false
  bool own_gpu_data_;
  // 为了支持多个设备
  int gpu_device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
