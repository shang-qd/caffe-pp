#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/CaffeCL.h"
#include "caffe/util/math_functions_cl.hpp"

namespace caffe {

static const char* cl_file = "./cl/SoftmaxLoss.cl";

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  if (Caffe::mode() == Caffe::CL) {
  	  CaffeCL *cl = CaffeCL::Instance();
  	  std::vector<string> vs = { "SoftmaxLossForward","SoftmaxLossBackward" };
  	  cl->CreateProgram(cl_file, vs);
   }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

void SoftmaxLossForward(int nthreads, float *prob_data, float *label, float *loss_data,
    int outer_num_, int dim, int inner_num_, int has_ignore_label_, int ignore_label_, float *counts)
{
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["SoftmaxLossForward"];
	clSetKernelArg(kernel, 0, sizeof(int), &nthreads);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &prob_data);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &label);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &loss_data);
	clSetKernelArg(kernel, 4, sizeof(int), &outer_num_);
	clSetKernelArg(kernel, 5, sizeof(int), &dim);
	clSetKernelArg(kernel, 6, sizeof(int), &inner_num_);
	clSetKernelArg(kernel, 7, sizeof(int), &has_ignore_label_);
	clSetKernelArg(kernel, 8, sizeof(int), &ignore_label_);
	clSetKernelArg(kernel, 9, sizeof(cl_mem), &counts);
	size_t g[1] = { (size_t)nthreads };
	size_t l[1]= { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cl(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
	  const Dtype* prob_data = prob_.gpu_data();
	  const Dtype* label = bottom[1]->gpu_data();
	  const int dim = prob_.count() / outer_num_;
	  const int nthreads = outer_num_ * inner_num_;
	  // Since this memory is not used for anything until it is overwritten
	  // on the backward pass, we use it here to avoid having to allocate new GPU
	  // memory to accumulate intermediate results in the kernel.
	  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
	  // Similarly, this memory is never used elsewhere, and thus we can use it
	  // to avoid having to allocate additional GPU memory.
	  Dtype* counts = prob_.mutable_gpu_diff();
	  // NOLINT_NEXT_LINE(whitespace/operators)

	  SoftmaxLossForward(nthreads, (float*)prob_data, (float*)label,
			  (float*)loss_data, outer_num_, dim, inner_num_, has_ignore_label_,
			  ignore_label_,(float*)counts);

	  Dtype loss;
	  math_cl::caffe_cl_asum(nthreads, (float*)loss_data, (float*)&loss);
	  Dtype valid_count = -1;
	  if (normalization_ == LossParameter_NormalizationMode_VALID &&
	      has_ignore_label_) {
		  math_cl::caffe_cl_asum(nthreads, (float*)counts, (float*)&valid_count);
	  }
	  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
	                                                        valid_count);
	  if (top.size() == 2) {
	    top[1]->ShareData(prob_);
	  }
}



void SoftmaxLossBackward(const int nthreads, const float* top,
          const float* label, float* bottom_diff, const int num, const int dim,
          const int spatial_dim, const int has_ignore_label_,
          const int ignore_label_,  float* counts)
{

	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["SoftmaxLossBackward"];
	clSetKernelArg(kernel, 0, sizeof(int), &nthreads);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &top);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &label);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &bottom_diff);
	clSetKernelArg(kernel, 4, sizeof(int), &num);
	clSetKernelArg(kernel, 5, sizeof(int), &dim);
	clSetKernelArg(kernel, 6, sizeof(int), &spatial_dim);
	clSetKernelArg(kernel, 7, sizeof(int), &has_ignore_label_);
	clSetKernelArg(kernel, 8, sizeof(int), &ignore_label_);
	clSetKernelArg(kernel, 9, sizeof(cl_mem), &counts);
	size_t g[1] = { (size_t)nthreads };
	size_t l[1]= { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cl(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[1]) {
	    LOG(FATAL) << this->type()
	               << " Layer cannot backpropagate to label inputs.";
	  }
	  if (propagate_down[0]) {
	    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	    const Dtype* prob_data = prob_.gpu_data();
	    const Dtype* top_data = top[0]->gpu_data();
	    math_cl::caffe_copy(prob_.count(), prob_data, bottom_diff);
	    const Dtype* label = bottom[1]->gpu_data();
	    const int dim = prob_.count() / outer_num_;
	    const int nthreads = outer_num_ * inner_num_;
	    Dtype* counts = prob_.mutable_gpu_diff();
	    SoftmaxLossBackward(nthreads, (float*)top_data, (float*)label, (float*)bottom_diff,
	        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, (float*)counts);

	    Dtype valid_count = -1;
	    if (normalization_ == LossParameter_NormalizationMode_VALID &&
	        has_ignore_label_) {
	      math_cl::caffe_cl_asum(nthreads, (float*)counts, (float*)&valid_count);
	    }
	    const Dtype loss_weight = top[0]->cpu_diff()[0] /
	                              get_normalizer(normalization_, valid_count);
	    math_cl::caffe_cl_scal(prob_.count(), (float)loss_weight,
	    		(float*)bottom_diff);
	  }
  }
#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
