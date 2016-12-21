#include "caffe/util/math_functions_cl.hpp"
#include "caffe/CaffeCL.h"
#include <glog/logging.h>

#include <vector>
#include <string>

//template <typename Dtype>
void math_cl::Caffe_Copy(const int N, const float* X, float* Y)
{
	if (X != Y)
	{
		CaffeCL *cl = CaffeCL::Instance();
		cl_kernel kernel = cl->GetKernel(cl_file)["Caffe_Copy"];
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &X);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &Y);
		size_t g[1] = { (size_t)N };
		size_t l[1];
		if (N > 128)
		{
			l[0] = 128;
		}
		else
		{
			l[0] = 1;
		}
		if (cl->ExecKernel(kernel, 1, g, l) == false)
		{
			LOG(FATAL) << "CaffeCL::Caffe_Copy";
		}

	}
}

//INSTANTIATE_CLASS(math_cl);
