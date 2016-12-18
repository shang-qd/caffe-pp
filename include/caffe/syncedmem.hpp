/**
 *  ����caffe��OpenCL��֧�֣����֧�ָ��Ƿ���GPU�޹�
 *  ��Ϊ�ܶ�CPUҲͬ��֧��OpenCL. CUDA��OpenCL����ͬʱʹ��. 
 *  �����ڱ����ʱ���Ƿ�ʹ����CPU_ONLY��Ȼ��ʾ֧��OpenCL.
 *  Ŀǰ�İ汾��֧�ֵ�GPU
 *  �޸����ߣ����춫,���䣺shang_qd@qq.com
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
	// �����OpenCLģʽ,ʹ��free,malloc ��ʽ�����ڴ棬��cpu_malloc_use_cuda_��Զ����false
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

	// �����OpenCLģʽ,ʹ��free,malloc ��ʽ�����ڴ棬��cpu_malloc_use_cuda_��Զ����false
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
  // ��Ϊcuda��OpenCL����ͬʱ֧�֣�GPUģʽ��gpu_ptr_ ��ʾcuda ָ��
  // ��CLģʽ�£���ʾopenCL ָ�룬����ı���ͬ��
  void* gpu_ptr_;

  size_t size_;
  SyncedHead head_;
  // cpu_data_ �����Ƿ��ڱ���������ģ��ڱ�����Ϊtrue,���ⲿ����false
  bool own_cpu_data_;
  // �����ʹ����������������������ڴ棬�Ǿ����ڴ�֧��CPU��GPU�ڴ湲��
  bool cpu_malloc_use_cuda_;
  // gpu_data_ �����Ƿ��ڱ���������ģ��ڱ�����Ϊtrue,���ⲿ����false
  bool own_gpu_data_;
  // Ϊ��֧�ֶ���豸
  int gpu_device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
