#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"
#include "caffe/CaffeCL.h"

namespace caffe {

static const char* cl_file = "./cl/ReLU.cl";

template <typename Dtype>
void ReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	if (Caffe::mode() == Caffe::CL) {
				CaffeCL *cl = CaffeCL::Instance();
				std::vector<string> vs = { "ReLU_Forward", "ReLU_Backward" };
				cl->CreateProgram(cl_file, vs);
	}
}
template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const int count = bottom[0]->count();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["ReLU_Forward"];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &bottom_data);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &top_data);
	clSetKernelArg(kernel, 2, sizeof(Dtype), (Dtype*)&negative_slope);
	size_t g[1] = { (size_t)count };
	size_t l[1];
	l[0] = count > 128 ? 128 : 1;
	cl->ExecKernel(kernel, 1, g, l);
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cl(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]){
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
		size_t g[1] = { (size_t)count };
		size_t l[1];
		l[0] = count >128 ? 128 : 1;
		cl->ExecKernel(kernel,1,g,l);
	}
}

#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
