#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_cl.hpp"

#include "caffe/CaffeCL.h"

namespace caffe {

static const char* cl_file = "./cl/Conv.cl";


struct ParamIm2Col
{
	int a;
};

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::conv_im2col_cl(const Dtype* data,int offD,
		Dtype* col_buff) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {

    		CaffeCL *cl = CaffeCL::Instance();
    		cl_kernel kernel = cl->GetKernel(cl_file)["im2col"];
    		//clSetKernelArg(kernel, 0, sizeof(cl_mem), &bottom_data);
    		//clSetKernelArg(kernel, 1, sizeof(cl_mem), &top_data);
    		//clSetKernelArg(kernel, 2, sizeof(Dtype), (Dtype*)&negative_slope);
    		size_t g[1] = { (size_t)count };
    		size_t l[1] = { (size_t)CAFFE_CL_NUM_THREADS };
    		//cl->ExecKernel(kernel, 1, g, l);

    	  /*
        im2col_cl(data,offD, conv_in_channels_,
            conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
            kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
            */
      } else {
    	  LOG(FATAL) << "not use force_nd_im2col_";
      }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::conv_col2im_cl(const Dtype* col_buff,
		Dtype* data,int offD) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
    	  /*
        col2im_cl(col_buff, conv_in_channels_,
            conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
            kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            dilation_.cpu_data()[0], dilation_.cpu_data()[1], data,offD);
            */
      } else {
    	  LOG(FATAL) << "not use force_nd_im2col_";
      }
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cl_gemm(
		const Dtype* input,int off_input,
		const Dtype* weights,int off_weights,
		Dtype* output,int off_output, bool skip_im2col) {

/*
	const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cl(input, off_input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
    off_input = 0;
  }

  for (int g = 0; g < group_; ++g) {
	int off_i = off_input + col_offset_ * g;
	int off_w = off_weights + weight_offset_ * g;
	int off_o = off_output + output_offset_ * g;

	math_cl::caffe_cl_gemm(clblasNoTrans, clblasNoTrans,
			conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_,
			(Dtype)1., weights,off_w,
			col_buff,off_i,(Dtype)0.,
			output, off_o);
  }*/
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cl_bias(Dtype* output,int off_output,
    const Dtype* bias,int off_bias){
/*
	math_cl::caffe_cl_gemm<Dtype>(clblasNoTrans, clblasNoTrans, num_output_,
	      out_spatial_dim_, 1, (Dtype)1.,
		  bias,off_bias,
		  bias_multiplier_.gpu_data(),0,(Dtype)1.,
		  output,off_output);
		  */
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cl_gemm(const Dtype* output,int off_output,
    const Dtype* weights,int off_weights, Dtype* input,int off_input)
	{

	/*
	 Dtype* col_buff = col_buffer_.mutable_gpu_data();
	  if (is_1x1_) {
	    col_buff = input;
	  }
	  for (int g = 0; g < group_; ++g) {
		  math_cl::caffe_cl_gemm<Dtype>(clblasTrans, clblasNoTrans, kernel_dim_,
				  conv_out_spatial_dim_, conv_out_channels_ / group_, (Dtype)1.,
				  weights,off_weights + weight_offset_ * g,
				  output,off_output + output_offset_ * g,(Dtype)0.,
				  col_buff,off_input + col_offset_ * g);
	  }
	  if (!is_1x1_) {
			conv_col2im_cl(col_buff, input,off_input);
	  }*/
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cl_gemm(const Dtype* input,int off_input,
    const Dtype* output,int off_output, Dtype* weights,int off_weights) {
/*
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cl(input,off_input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
    off_input = 0;
  }
  for (int g = 0; g < group_; ++g) {
	  math_cl::caffe_cl_gemm<Dtype>(clblasNoTrans, clblasTrans, conv_out_channels_ / group_,
			  kernel_dim_, conv_out_spatial_dim_, (Dtype)1.,
			  output, off_output + output_offset_ * g,
			  col_buff,off_input + col_offset_ * g, (Dtype)1.,
			  weights,off_weights + weight_offset_ * g);
   }*/
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cl_bias(Dtype* bias,int off_bias,
    const Dtype* input,int off_input) {
	/*
	math_cl::caffe_cl_gemv<Dtype>(clblasNoTrans,num_output_,out_spatial_dim_,(Dtype)1.,
			input,off_input,
			bias_multiplier_.gpu_data(),0,(Dtype)1.,
			bias,off_bias);*/
}

}
