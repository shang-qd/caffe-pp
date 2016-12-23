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
		cl_kernel kernel = cl->GetKernel(cl_file)["caffe_copy"];
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &X);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &Y);
		size_t g[1] = { (size_t)N };
		size_t l[1];
		l[0] = N > 128 ? 128 : 1;
		cl->ExecKernel(kernel, 1, g, l);
	}
}


template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <typename Dtype>
void caffe_cl_gemm(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A,size_t offA, const Dtype* B,size_t offB,
	const Dtype beta, Dtype* C,size_t offC)
{

	int lda = (TransA == clblasNoTrans) ? K : M;
	int ldb = (TransB == clblasNoTrans) ? N : K;
	CaffeCL *cl = CaffeCL::Instance();

	//LOG(INFO) << offA << ":" << offB << ":" << offC;
	cl_event event = NULL;
	cl_int err = clblasSgemm(clblasRowMajor, TransB, TransA,
				N, M, K, (float)alpha,
				(cl_mem)A, offA, lda,
				(cl_mem)B, offB, ldb, (float)beta,
				(cl_mem)C, offC, N,
				1, &cl->m_commandQueue, 0, NULL, &event);
	err = clWaitForEvents(1, &event);
	if (err != CL_SUCCESS) {
		LOG(FATAL) << "caffe_cl_gemm" << err;
	}
}

template void caffe_cl_gemm<float>(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int N, const int K,
    const float alpha, const float* A,size_t offA, const float* B,size_t offB,
	const float beta,float* C,size_t offC);

template void caffe_cl_gemm<double>(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int N, const int K,
    const double alpha, const double* A,size_t offA, const double* B,size_t offB,
	const double beta,double* C,size_t offC);


template <typename Dtype>
void caffe_cl_gemv(const clblasTranspose TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A,size_t offA, const Dtype* x,size_t offX, const Dtype beta,
    Dtype* y,size_t offY)
{
	CaffeCL *cl = CaffeCL::Instance();
	cl_int err = clblasSgemv(clblasRowMajor, TransA, N, M, (float)alpha,
			(cl_mem)A, offA, N,
			(cl_mem)x, offX, 1, (float)beta,
			(cl_mem)y, offY, 1,
			1, &cl->m_commandQueue, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		LOG(FATAL) << "caffe_cl_gemv" << err;
	}
}


template void caffe_cl_gemv<float>(const clblasTranspose TransA, const int M, const int N,
    const float alpha, const float* A,size_t offA, const float* x,size_t offX, const float beta,
	float* y,size_t offY);

template void caffe_cl_gemv<double>(const clblasTranspose TransA, const int M, const int N,
    const double alpha, const double* A,size_t offA, const double* x,size_t offX, const double beta,
	double* y,size_t offY);

}; // end namespace
