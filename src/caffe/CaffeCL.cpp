#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "caffe/CaffeCL.h"
#include "caffe/util/math_functions_cl.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <glog/logging.h>

const char* cl_file = "./cl/cl_common.cl";

CaffeCL::CaffeCL() : m_context(nullptr), m_commandQueue(nullptr), m_device(nullptr)
{
	if (CreateContext() == false)
	{
		LOG(FATAL) << "CaffeCL";
		return;
	}
	if (CreateCommandQueue() == false)
	{
		LOG(FATAL) << "CaffeCL";
		return;
	}
	Init();
}

bool CaffeCL::Init()
{
	std::vector<std::string> kn = {"caffe_copy",
			"im2col","im2col_nd",
			"col2im","col2im_nd"};
	CreateProgram(cl_file,kn);
	cl_int err = clblasSetup();
	if (err != CL_SUCCESS) {
		LOG(FATAL) << "clblasSetup";
	}
	return true;
}

CaffeCL::~CaffeCL()
{
	if (m_commandQueue != nullptr)
	{
		clReleaseCommandQueue(m_commandQueue);
	}
	if (m_context != nullptr)
	{
		clReleaseContext(m_context);
	}
}

bool CaffeCL::CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;

	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		LOG(FATAL) << "CaffeCL::CreateContext" << errNum;
		return false;
	}

	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	m_context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		LOG(INFO) << "Could not create GPU context, trying CPU...";
		m_context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
			NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			LOG(FATAL) << "CaffeCL::CreateContext" << errNum;
			return false;
		}
	}
	return true;
}

bool CaffeCL::CreateCommandQueue()
{
	cl_int errNum;
	cl_device_id *devices;
	size_t deviceBufferSize = -1;
	errNum = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		LOG(FATAL) << "CaffeCL::CreateCommandQueue";
		return false;
	}

	if (deviceBufferSize <= 0)
	{
		LOG(FATAL) << "CaffeCL::CreateCommandQueue" << errNum;
		return false;
	}

	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete[] devices;
		LOG(FATAL) << "CaffeCL::CreateCommandQueue " << errNum;
		return false;
	}

	m_commandQueue = clCreateCommandQueue(m_context, devices[0], CL_QUEUE_PROFILING_ENABLE, &errNum);
	if (m_commandQueue == nullptr)
	{
		delete[] devices;
		LOG(FATAL) << "CaffeCL::CreateCommandQueue " << errNum;
		return false;
	}

	m_device = devices[0];
	delete[] devices;
	return true;
}

bool CaffeCL::CreateProgram(const char* fileName, const vector<string> &vsk)
{
	if (m_programs.find(fileName) != m_programs.end())
	{
		return true;
	}
	cl_int err = CL_SUCCESS;
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		LOG(FATAL) << "CaffeCL::CreateProgram" << fileName;
		return false;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	cl_program program = clCreateProgramWithSource(m_context, 1,
		(const char**)&srcStr,NULL, &err);
	if (program == nullptr)
	{
		LOG(FATAL) << "CaffeCL::CreateProgram" << err;
		return false;
	}
	cl_int errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		char buildLog[16384];
		clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG,sizeof(buildLog), buildLog, NULL);

		LOG(FATAL) << "CaffeCL::CreateProgram:" << buildLog;
		return false;
	}
	program_info pi;
	pi.program = program;
	map<string, cl_kernel> kernels;
	for (size_t i = 0; i < vsk.size(); ++i)
	{
		cl_kernel k = clCreateKernel(program, vsk[i].c_str(), nullptr);
		if (k == nullptr)
		{
			LOG(FATAL) << "CaffeCL::CreateProgram" << vsk[i];
		}
		kernels.insert(std::make_pair(vsk[i], k));
	}
	pi.kernels = kernels;
	m_programs.insert(std::make_pair(fileName, pi));
	return true;
}

cl_mem CaffeCL::CreateReadMem(void *d, int size)
{
	cl_mem mem;
	cl_int err = CL_SUCCESS;
	mem = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, d, &err);
	if (err != CL_SUCCESS)
	{
		LOG(FATAL) << "CaffeCL::CreateReadMem" << err;
	}
	return mem;
}

cl_mem CaffeCL::CreateWriteMem(void *d, int size)
{
	cl_mem mem;
	cl_int err = CL_SUCCESS;
	if (d == nullptr)
	{
		mem = clCreateBuffer(m_context, CL_MEM_READ_WRITE, size, nullptr, &err);
	}
	else
	{
		mem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, d, &err);
	}
	if (err != CL_SUCCESS)
	{
		LOG(FATAL) << "CaffeCL::CreateWriteMem" << err;
	}
	return mem;
}

bool CaffeCL::ExecKernel(cl_kernel kernel, int dim, const size_t *g, const size_t *l)
{
	cl_int res = clEnqueueNDRangeKernel(m_commandQueue, kernel, dim, nullptr, g, l, 0, nullptr, nullptr);
	
	if (res != CL_SUCCESS)
	{
		LOG(FATAL) << "CaffeCL::ExecKernel" << res;
	}
	return true;
}

void CaffeCL::gpu2host(int size, void *gpu_ptr, void*cpu_ptr)
{
	cl_int res = clEnqueueReadBuffer(m_commandQueue, (cl_mem)gpu_ptr, CL_TRUE, 0, size, cpu_ptr, 0, nullptr, nullptr);
	if (res != CL_SUCCESS)
	{
		LOG(FATAL) << "CaffeCL::gpu2host" << res;
	}
}

bool CaffeCL::read_buf(cl_mem mem, int size, void* p)
{
	//printf("mem = %p; \n", mem);
	//printf("size = %d; \n", size);
	//printf("p = %p; \n", p);
	//cl_int res = clEnqueueReadBuffer(m_commandQueue, mem, CL_TRUE, 0, size, p, 0, nullptr, nullptr);
	return true;
}

//
void CaffeCL::host2gpu(int size, void *gpu_ptr, void*cpu_ptr)
{
	cl_int res = clEnqueueWriteBuffer(m_commandQueue, (cl_mem)gpu_ptr, CL_TRUE, 0, size, cpu_ptr, 0, nullptr, nullptr);
	if (res != CL_SUCCESS)
	{
		LOG(FATAL) << "CaffeCL::host2gpu" << res;
	}
}

map<string, cl_kernel> CaffeCL::GetKernel(string file)
{
	return m_programs[file].kernels;
}

CaffeCL* CaffeCL::Instance()
{
	static CaffeCL* instance = nullptr;
	if (instance == nullptr)
	{
		instance = new CaffeCL();
	}
	return instance;
}

void CaffeCL::Test(cl_kernel kernel)
{
	/*
	int ARRAY_SIZE = 128;
	float *a= new float[ARRAY_SIZE];
	float *b = new float[ARRAY_SIZE];
	float *res = new float[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		a[i] = i - 10;
		b[i] = a[i];
	}

	cl_mem cl_a = CreateWriteMem(a, ARRAY_SIZE * 4);
	//cl_mem cl_b = CreateWriteMem(b, ARRAY_SIZE * 4);
	//cl_mem cl_res = CreateWriteMem(nullptr, ARRAY_SIZE * 4);
	float f = 0;// new float;

	cl_int errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_a);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_a);
	errNum |= clSetKernelArg(kernel, 2, sizeof(float), &f);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error setting kernel arguments." << std::endl;
	}

	size_t g[1] = { ARRAY_SIZE };
	size_t l[1] = { 128 };

	errNum = clEnqueueNDRangeKernel(m_commandQueue, kernel, 1, nullptr, g, l, 0, nullptr, nullptr);

	errNum = clEnqueueReadBuffer(m_commandQueue, cl_a, CL_TRUE,0, ARRAY_SIZE * sizeof(float), res,0, NULL, NULL);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		std::cout << res[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Executed program succesfully." << std::endl;
	*/
}


void im2col_nd_gpu_kernel(const int num_axes,const int n, const float* data_im,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col)
{

}
