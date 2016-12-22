#ifndef CAFFE_UTIL_MATH_FUNCTIONS_CL_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_CL_H_


#include "caffe/common.hpp"
#include <clBLAS.h>

namespace math_cl{

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y);

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_cl_gemm(const clblasTranspose TransA,
    const clblasTranspose TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);


template <typename Dtype>
void caffe_cl_gemv(const clblasTranspose TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);
};

#endif
