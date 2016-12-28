#include "caffe/util/math_functions_cl.hpp"
#include "caffe/CaffeCL.h"
#include <glog/logging.h>

#include <mutex>
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


void cl_gemm_test(int M,int N,int K,
		float *col_buff,int off_a,
		float *weights, int off_b,
		float *output,int off_c)
{
	CaffeCL *cl = CaffeCL::Instance();
	cl_event event = NULL;
	clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
	          	      M, N, K,1,
					  (cl_mem)col_buff, off_a, N,
					  (cl_mem)weights, off_b, K, 0,
					  (cl_mem)output, off_c, K,
					  1, &cl->m_commandQueue, 0, NULL, &event);
	clWaitForEvents(1, &event);
}

template <typename Dtype>
void caffe_cl_gemm(const clblasTranspose TransA,const clblasTranspose TransB,
		const int M, const int N, const int K,const Dtype alpha,
		const Dtype *buf_a,int off_a,
		const Dtype *buf_b,int off_b,const Dtype beta,
		Dtype *buf_c,int off_c)
{
	int lda = (TransA == clblasNoTrans) ? K : M;
	int ldb = (TransB == clblasNoTrans) ? N : K;
	cl_int res = CL_SUCCESS;
	cl_event event = NULL;
	CaffeCL *cl = CaffeCL::Instance();
	res = clblasSgemm(clblasRowMajor, TransA, TransB,
				M, N, K, alpha,
				(cl_mem)buf_a,off_a, lda,
				(cl_mem)buf_b,off_b, ldb, beta,
				(cl_mem)buf_c,off_c, N,
				1, &cl->m_commandQueue, 0, NULL, &event);

	res |= clWaitForEvents(1, &event);
/*
	float *cpu_a = new float[M * K];
	float *cpu_b = new float[K * N];
	float *cpu_c = new float[M * N];

	res |= clEnqueueReadBuffer(cl->m_commandQueue, (cl_mem)buf_a, CL_TRUE, 0,
							M * K * 4,cpu_a, 0, nullptr, nullptr);
	res |= clEnqueueReadBuffer(cl->m_commandQueue, (cl_mem)buf_b, CL_TRUE, 0,
							K * N * 4,cpu_b, 0, nullptr, nullptr);

	res |= clEnqueueReadBuffer(cl->m_commandQueue, (cl_mem)buf_c, CL_TRUE, 0,
				M * N * 4,cpu_c, 0, nullptr, nullptr);

	float sum_a = 0;
	float sum_b = 0;
	float sum_c = 0;

	float d_sum_c = 0;
	  float *d_c = new float[M * N];
	  for (int i = 0; i < M; i++){
		  for (int j = 0; j < N; j++){
			  float sum = 0;
			  for (int k = 0; k < K; k++){
				  sum += cpu_a[i * K + k] * cpu_b[k * N + j];
			  }
			  d_c[i * N + j] = sum;
		  }
	  }
	for (int i = 0; i < M * K; i++){
		  sum_a += cpu_a[i];
	  }
	  for (int i = 0; i < K * N; i++){
		  sum_b += cpu_b[i];
	  }
	  for (int i = 0; i < M * N; i++){
		  sum_c += cpu_c[i];
		  d_sum_c += d_c[i];
	  }
	  LOG(INFO) << M << " : " << N << " : " << K;
	  LOG(FATAL) << "CL " << sum_a << ":" <<  sum_b << ":" << sum_c << " = " << d_sum_c;
*/
	if (res != CL_SUCCESS) {
			LOG(FATAL) << "CaffeCL::gpu2host" << res;
	}
}

template void caffe_cl_gemm<float>(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int K, const int N,
    const float alpha, const float* A,int offA, const float* B,int offB,
	const float beta,float* C,int offC);

template void caffe_cl_gemm<double>(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int K, const int N,
    const double alpha, const double* A,int offA, const double* B,int offB,
	const double beta,double* C,int offC);

template <typename Dtype>
void caffe_cl_gemv(const clblasTranspose TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A,int offA, const Dtype* x,int offX, const Dtype beta,
    Dtype* y,int offY) {
	// TODO gemv:use gemm 替代
	caffe_cl_gemm(TransA,clblasNoTrans,M,1,N,alpha,A,offA,x,offX,beta,y,offY);
}

template void caffe_cl_gemv<float>(const clblasTranspose TransA, const int M, const int N,
    const float alpha, const float* A,int offA, const float* x,int offX, const float beta,
	float* y,int offY);

template void caffe_cl_gemv<double>(const clblasTranspose TransA, const int M, const int N,
    const double alpha, const double* A,int offA, const double* x,int offX, const double beta,
	double* y,int offY);
}; // end namespace
