#ifndef CAFFE_UTIL_MATH_FUNCTIONS_CL_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_CL_H_


#include "caffe/common.hpp"
#include <clBLAS.h>

namespace math_cl{


template <typename Dtype>
Dtype debug_sum(const Dtype* add,int n);

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y);

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

void caffe_cl_mul(const int N, const float* a,
    const float* b, float* y);

void caffe_cl_asum(const int n, const float* x, float* y);

void caffe_cl_scal(const int N, const float alpha, float *X);

};


#endif
