#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/CaffeCL.h"
#include "caffe/util/math_functions_cl.hpp"

namespace caffe {

static const char* cl_file = "./cl/Softmax.cl";

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
  if (Caffe::mode() == Caffe::CL) {
	  CaffeCL *cl = CaffeCL::Instance();
	  std::vector<string> vs = { "channel_max", "channel_subtract",
			  "channel_exp","channel_sum","channel_div","channel_dot" };
	  cl->CreateProgram(cl_file, vs);
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

void channel_max(int outer_num_,int channels,int inner_num_,
		float *top_data,float *scale_data)
{
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["channel_max"];
	clSetKernelArg(kernel, 0, sizeof(int), &outer_num_);
	clSetKernelArg(kernel, 1, sizeof(int), &channels);
	clSetKernelArg(kernel, 2, sizeof(int), &inner_num_);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &top_data);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &scale_data);
	size_t g[1] = { (size_t)(outer_num_ * inner_num_) };
	size_t l[1]= { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);

}

void channel_subtract(int count,int outer_num_,int channels,int inner_num_,
	float *scale_data,float *top_data)
{
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["channel_subtract"];
	clSetKernelArg(kernel, 0, sizeof(int), &count);
	clSetKernelArg(kernel, 1, sizeof(int), &outer_num_);
	clSetKernelArg(kernel, 2, sizeof(int), &channels);
	clSetKernelArg(kernel, 3, sizeof(int), &inner_num_);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &scale_data);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &top_data);
	size_t g[1] = { (size_t)count };
	size_t l[1]= { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

void channel_exp(int count,float *top_data)
{
	//kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	//      count, top_data, top_data);
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["channel_exp"];
	clSetKernelArg(kernel, 0, sizeof(int), &count);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &top_data);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &top_data);
	size_t g[1] = { (size_t)count };
	size_t l[1]= { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

void channel_sum(int outer_num_,int channels,int inner_num_,
		float *top_data,float *scale_data)
{
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["channel_sum"];
	clSetKernelArg(kernel, 0, sizeof(int), &outer_num_);
	clSetKernelArg(kernel, 1, sizeof(int), &channels);
	clSetKernelArg(kernel, 2, sizeof(int), &inner_num_);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &top_data);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &scale_data);
	size_t g[1] = { (size_t)outer_num_ * inner_num_ };
	size_t l[1]= { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

void channel_div(int count,int outer_num_,int channels,int inner_num_,
		float *scale_data,float *top_data)
{
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["channel_div"];
	clSetKernelArg(kernel, 0, sizeof(int), &count);
	clSetKernelArg(kernel, 1, sizeof(int), &outer_num_);
	clSetKernelArg(kernel, 2, sizeof(int), &channels);
	clSetKernelArg(kernel, 3, sizeof(int), &inner_num_);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &scale_data);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &top_data);
	size_t g[1] = { (size_t)count };
	size_t l[1]= { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);

}

void channel_dot(int outer_num_, int channels, int inner_num_,
		      float *top_diff, float *top_data, float *scale_data)
{
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["channel_dot"];
	clSetKernelArg(kernel, 1, sizeof(int), &outer_num_);
	clSetKernelArg(kernel, 2, sizeof(int), &channels);
	clSetKernelArg(kernel, 3, sizeof(int), &inner_num_);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &top_diff);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &top_data);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &scale_data);
	size_t g[1] = { (size_t)outer_num_ * inner_num_ };
	size_t l[1]= { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->gpu_data();
	  Dtype* top_data = top[0]->mutable_gpu_data();
	  Dtype* scale_data = scale_.mutable_gpu_data();
	  int count = bottom[0]->count();
	  int channels = top[0]->shape(softmax_axis_);
	  //caffe_copy(count, bottom_data, top_data);
	  math_cl::caffe_copy(count, bottom_data, top_data);

	  channel_max(outer_num_,channels,inner_num_,
			  (float*)top_data, (float*)scale_data);

	  channel_subtract(count,outer_num_,channels,inner_num_,
			  (float*)scale_data,(float*)top_data);
	  channel_exp(count,(float*)top_data);

	  channel_sum(outer_num_,channels,inner_num_,
	  		(float*)top_data,(float*)scale_data);

	  channel_div(count,outer_num_,channels,inner_num_,
	  		(float*)scale_data,(float*)top_data);
}


template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cl(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	 const Dtype* top_diff = top[0]->gpu_diff();
	  const Dtype* top_data = top[0]->gpu_data();
	  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	  Dtype* scale_data = scale_.mutable_gpu_data();
	  int count = top[0]->count();
	  int channels = top[0]->shape(softmax_axis_);
	  math_cl::caffe_copy(count, top_diff, bottom_diff);

	  channel_dot(outer_num_, channels, inner_num_,
	  		      (float*)top_diff, (float*)top_data, (float*)scale_data);

	  channel_subtract(count, outer_num_, channels, inner_num_,
	  	      (float*)scale_data, (float*)bottom_diff);

	  math_cl::caffe_cl_mul(top[0]->count(), (float*)bottom_diff,
			  (float*)top_data, (float*)bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
