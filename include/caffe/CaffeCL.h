#pragma once

#include <CL/cl.h>
#include <string>
#include <vector>
#include <map>

using namespace std;

extern const char* cl_file;// = "cl_common.cl";
//
class CaffeCL
{
	struct program_info
	{
		cl_program program;
		map<string, cl_kernel> kernels;
	};
public:
	cl_context m_context;
	cl_command_queue m_commandQueue;
	cl_device_id m_device;
	map<string, program_info> m_programs;

	CaffeCL();
	//
	bool Init();
	bool CreateContext();
	bool CreateCommandQueue();

	map<string, cl_kernel> GetKernel(string file);

	bool ExecKernel(cl_kernel kernel, int dim, const size_t *g, const size_t *l);

	bool CreateProgram(const char* fileName, const vector<string> &vsk);
	cl_mem CreateReadMem(void *d, int size);
	cl_mem CreateWriteMem(void *d, int size);
	
	void gpu2host(int size, void *gpu_ptr, void*cpu_ptr);

	void host2gpu(int size, void *gpu_ptr, void*cpu_ptr);
	
	bool read_buf(cl_mem mem, int size, void* p);
	static CaffeCL* Instance();

	~CaffeCL();
	
	void Test(cl_kernel kernel);
};

