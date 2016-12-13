
#include "caffe/CaffeCL.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

CaffeCL::CaffeCL() : m_context(nullptr), m_commandQueue(nullptr), m_device(nullptr)
{
	if (CreateContext() == false)
	{
		std::cerr << "Failed to create OpenCL context." << std::endl;
		return;
	}
	if (CreateCommandQueue() == false)
	{
		std::cerr << "Failed to create OpenCL CommandQueue." << std::endl;
		return;
	}
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
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return nullptr;
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
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		m_context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
			NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
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
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return false;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return false;
	}

	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete[] devices;
		std::cerr << "Failed to get device IDs";
		return false;
	}

	m_commandQueue = clCreateCommandQueue(m_context, devices[0], 0, NULL);
	if (m_commandQueue == nullptr)
	{
		delete[] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return false;
	}

	m_device = devices[0];
	delete[] devices;
	return true;
}

bool CaffeCL::CreateProgram(const char* fileName, const vector<string> &vsk)
{
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		throw "没有找到文件";
		return false;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	cl_program program = clCreateProgramWithSource(m_context, 1,
		(const char**)&srcStr,NULL, NULL);
	if (program == nullptr)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return false;
	}

	cl_int errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		char buildLog[16384];
		clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG,sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
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
			std::cerr << "Failed to create kernel" << std::endl;
		}
		kernels.insert(std::make_pair(vsk[i], k));
	}
	pi.kernels = kernels;
	m_programs.insert(std::make_pair(fileName, pi));
	return true;
}

cl_mem CaffeCL::CreateReadMem(void *d, int size)
{
	return clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, d, nullptr);
}

cl_mem CaffeCL::CreateWriteMem(int size)
{
	return clCreateBuffer(m_context, CL_MEM_READ_WRITE,size, nullptr, nullptr);
}

/*
void CaffeCL::Sample(const string &k)
{
int  ARRAY_SIZE = 128;
float *result = new float[ARRAY_SIZE];
float *a = new float[ARRAY_SIZE];
float *b = new float[ARRAY_SIZE];
for (int i = 0; i < ARRAY_SIZE; i++)
{
a[i] = (float)i;
b[i] = (float)(i * 2);
}
cl_kernel kernel = kernels[k];
cl_mem mem_a = CreateReadMem(a,ARRAY_SIZE);
cl_mem mem_b = CreateReadMem(b, ARRAY_SIZE);
cl_mem mem_res = CreateWriteMem(ARRAY_SIZE);

clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_a);
clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_b);
clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_res);

size_t globalWorkSize[1] = { ARRAY_SIZE };
size_t localWorkSize[1] = { 128 };

errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr,
globalWorkSize, localWorkSize, 0, nullptr, nullptr);

clEnqueueReadBuffer(commandQueue, mem_res, CL_TRUE,
0, ARRAY_SIZE * sizeof(float), result,0, nullptr, nullptr);

clReleaseMemObject(mem_a);
clReleaseMemObject(mem_b);
clReleaseMemObject(mem_res);

for (int i = 0; i < ARRAY_SIZE; i++)
{
std::cout << result[i] << " ";
}
std::cout << std::endl;
std::cout << "Executed program succesfully." << std::endl;
}
*/
