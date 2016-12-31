#include "caffe/util/math_functions_cl.hpp"
#include "caffe/CaffeCL.h"
#include <glog/logging.h>

#include <mutex>
#include <vector>
#include <string>

namespace math_cl{

template <typename Dtype>
Dtype debug_sum(const Dtype* add,int n)
{
	Dtype res = (Dtype)0.;
	for (int i = 0; i < n; i++){
		res += add[i];
	}
	return res;
}

template float debug_sum<float>(const float* add,int n);
template double debug_sum<double>(const double* add,int n);


template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y)
{
	if (X != Y) {
		CaffeCL *cl = CaffeCL::Instance();
		cl_kernel kernel = cl->GetKernel(cl_file)["caffe_copy"];
		clSetKernelArg(kernel, 0, sizeof(int), &N);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &X);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &Y);
		size_t g[1] = { (size_t)N };
		size_t l[1] = { (size_t)CAFFE_CL_NUM_THREADS };
		cl->ExecKernel(kernel, 1, g, l);
	}
}


template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {

	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["caffe_set"];
	clSetKernelArg(kernel, 0, sizeof(int), &N);
	clSetKernelArg(kernel, 1, sizeof(float), &alpha);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &Y);
	size_t g[1] = { (size_t)N };
	size_t l[1] = { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

//template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);


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
	if (res != CL_SUCCESS) {
		LOG(FATAL) << "clblasSgemm: " << res;
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
	caffe_cl_gemm(TransA,clblasNoTrans,N,1,M,alpha,A,offA,x,offX,beta,y,offY);
}

template void caffe_cl_gemv<float>(const clblasTranspose TransA, const int M, const int N,
    const float alpha, const float* A,int offA, const float* x,int offX, const float beta,
	float* y,int offY);

template void caffe_cl_gemv<double>(const clblasTranspose TransA, const int M, const int N,
    const double alpha, const double* A,int offA, const double* x,int offX, const double beta,
	double* y,int offY);

void caffe_cl_mul(const int N, const float* a,
		const float* b, float* y) {
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["caffe_mul"];
	clSetKernelArg(kernel, 0, sizeof(int), &N);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &y);
	size_t g[1] = { (size_t)N };
	size_t l[1] = { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

void caffe_cl_asum(const int n, const float* x, float* y) {
	// TODO 方法很笨
    float *res = new float[n];
    CaffeCL *cl = CaffeCL::Instance();
    clEnqueueReadBuffer(cl->m_commandQueue, (cl_mem)x, CL_TRUE, 0,
    		n * sizeof(cl_float),res, 0, NULL, NULL);
    for (int i = 0; i < n; i++) {
    	*y += res[i];
    }
    delete []res;
}


void caffe_cl_scal(const int N, const float alpha, float *X) {
  //CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
	CaffeCL *cl = CaffeCL::Instance();
	cl_kernel kernel = cl->GetKernel(cl_file)["caffe_scal"];
	clSetKernelArg(kernel, 0, sizeof(int), &N);
	clSetKernelArg(kernel, 1, sizeof(float), &alpha);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &X);
	size_t g[1] = { (size_t)N };
	size_t l[1] = { (size_t)CAFFE_CL_NUM_THREADS };
	cl->ExecKernel(kernel, 1, g, l);
}

}; // end namespace
