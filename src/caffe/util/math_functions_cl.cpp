#include "caffe/util/math_functions_cl.hpp"
#include "caffe/CaffeCL.h"
#include <glog/logging.h>

#include <vector>
#include <string>


namespace math_cl{

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y)
{
	if (X != Y) {
		CaffeCL *cl = CaffeCL::Instance();
		cl_kernel kernel = cl->GetKernel(cl_file)["Caffe_Copy"];
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &X);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &Y);
		size_t g[1] = { (size_t)N };
		size_t l[1];
		if (N > 128) {
			l[0] = 128;
		} else {
			l[0] = 1;
		}
		if (cl->ExecKernel(kernel, 1, g, l) == false) {
			LOG(FATAL) << "CaffeCL::Caffe_Copy";
		}
	}
}

template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <typename Dtype>
void caffe_cl_gemm(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C)
{
	throw "abc";
}

template void caffe_cl_gemm<float>(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);

template void caffe_cl_gemm<double>(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
	double* C);



template <typename Dtype>
void caffe_cl_gemv(const clblasTranspose TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y)
{
	throw "abc";
}


template void caffe_cl_gemv<float>(const clblasTranspose TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta,
	float* y);

template void caffe_cl_gemv<double>(const clblasTranspose TransA, const int M, const int N,
    const double alpha, const double* A, const double* x, const double beta,
	double* y);


}; // end namespace
