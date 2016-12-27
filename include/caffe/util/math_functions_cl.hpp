#ifndef CAFFE_UTIL_MATH_FUNCTIONS_CL_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_CL_H_


#include "caffe/common.hpp"
#include <clBLAS.h>

namespace math_cl{


void cl_gemm_test(int M,int N,int K,
		float *col_buff,int off_a,
		float *weights, int off_b,
		float *output,int off_c);

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y);

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_cl_gemm(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A,int offA, const Dtype* B,int offB,
	const Dtype beta, Dtype* C,int offC);


template <typename Dtype>
void caffe_cl_gemv(const clblasTranspose TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A,int offA, const Dtype* x,int offX, const Dtype beta,
    Dtype* y,int offY);
};

#endif
