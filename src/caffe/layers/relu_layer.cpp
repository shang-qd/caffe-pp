#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

#include "../CaffeCL.h"

namespace caffe {

	static const char* cl_file = "ReLU.cl";

	// 加载本层的程序对象和内核对象
	template <typename Dtype>
	void ReLULayer<Dtype>::InitCL(CaffeCL *cl)
	{
		std::vector<string> vs = { "ReLU_Forward", "ReLU_Backward" };
		cl->CreateProgram(cl_file, vs);
	}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cl(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

	CaffeCL *cl = CaffeCL::Instance();

	cl_kernel kernel = cl->GetKernel(cl_file)["ReLU_Forward"];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (cl_mem*)&bottom_data);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (cl_mem)&top_data);
	clSetKernelArg(kernel, 2, sizeof(Dtype), &negative_slope);

	size_t g[1] = { count };
	size_t l[1] = { 128 };

	cl->ExecKernel(kernel,1,g,l);
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  
  for (int i = 0; i < count; ++i) 
  {
	  if (bottom_data[i] > 0)
	  {
		  top_data[i] = bottom_data[i];
	  }
	  else
	  {
		  top_data[i] = negative_slope * bottom_data[i];
	  }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cl(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom)
{
	if (propagate_down[0])
	{
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
		
		CaffeCL *cl = CaffeCL::Instance();

		cl_kernel kernel = cl->GetKernel(cl_file)["ReLU_Backward"];

		clSetKernelArg(kernel, 0, sizeof(cl_mem), (cl_mem*)&bottom_data);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), (cl_mem*)&top_diff);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), (cl_mem*)&bottom_diff);
		clSetKernelArg(kernel, 3, sizeof(float), (cl_mem*)&negative_slope);

		size_t g[1] = { count };
		size_t l[1] = { 128 };

		cl->ExecKernel(kernel,1,g,l);
	}
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[0]) 
  {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
	
	for (int i = 0; i < count; ++i) 
	{
		if (bottom_data[i] > 0)
		{
			bottom_diff[i] = top_diff[i];
		}
		else
		{
			bottom_diff[i] = negative_slope * top_diff[i];
		}
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
