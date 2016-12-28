#include <stdio.h>
#include <sys/types.h>

	/* Include the clBLAS header. It includes the appropriate OpenCL headers */
#include <clBLAS.h>

//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
#include <CL/cl.h>

int clblas_test(void);

static const clblasOrder order = clblasRowMajor;
static const size_t M = 4;
static const size_t N = 5;
static const cl_float alpha = 10;
static const clblasTranspose transA = clblasNoTrans;
static const cl_float A[] = {
	11, 12, 13, 14, 15,
	21, 22, 23, 24, 25,
	31, 32, 33, 34, 35,
	41, 42, 43, 44, 45
};
static const size_t lda = 5;        /* i.e. lda = N */
static const cl_float X[] = {
	11,
	21,
	31,
	41,
	51,
};
static const int incx = 1;
static const cl_float beta = 20;
static cl_float Y[] = {
	12,
	21,
	31,
	41,
};
static const int incy = 1;
static cl_float result[4];          /* M */
static const size_t off = 1;
static const size_t offA = 5 + 1;   /* M + off */
static const size_t offX = 1;       /* off */
static const size_t offY = 1;       /* off */
static void
printResult(const char* str)
{
	size_t i, nElements;
	printf("%s:\n", str);
	nElements = (sizeof(result) / sizeof(cl_float)) / incy;
	for (i = 0; i < nElements; i++) {
		printf("%d\n", (int)result[i * incy]);
	}
}

int clblas_test(void)
{
	cl_int err;
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context ctx = 0;
	cl_command_queue queue = 0;
	cl_mem bufA, bufX, bufY;
	cl_event event = NULL;
	int ret = 0;
	/* Setup OpenCL environment. */
	/*
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS) {
		printf("clGetPlatformIDs() failed with %d\n", err);
		return 1;
	}
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		printf("clGetDeviceIDs() failed with %d\n", err);
		return 1;
	}
	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("clCreateContext() failed with %d\n", err);
		return 1;
	}
	queue = clCreateCommandQueue(ctx, device, 0, &err);
	if (err != CL_SUCCESS) {
		printf("clCreateCommandQueue() failed with %d\n", err);
		clReleaseContext(ctx);
		return 1;
	}*/

	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
	queue = clCreateCommandQueue(ctx, device, 0, &err);


	/* Setup clblas. */
	err = clblasSetup();
	if (err != CL_SUCCESS) {
		printf("clblasSetup() failed with %d\n", err);
		clReleaseCommandQueue(queue);
		clReleaseContext(ctx);
		return 1;
	}
	/* Prepare OpenCL memory objects and place matrices inside them. */
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * N * sizeof(*A) * 2,
		NULL, &err);
	bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(*X),
		NULL, &err);
	bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * sizeof(*Y),
		NULL, &err);
	err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
		M * N * sizeof(*A) * 2, A, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
		N * sizeof(*X), X, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0,
		M * sizeof(*Y), Y, 0, NULL, NULL);
	/* Call clblas extended function. */
	//float alpha_ = 1;
	//float beta_ = 0;    
	err = clblasSgemv(order, transA, M - off, N - off, alpha,
		bufA, offA, lda, bufX, offX, incx, beta,
		bufY, offY, incy, 1, &queue, 0, NULL, &event);
	if (err != CL_SUCCESS) {
		printf("clblasSgemvEx() failed with %d\n", err);
		ret = 1;
	}
	 else {
		/* Wait for calculations to be finished. */
		err = clWaitForEvents(1, &event);
		/* Fetch results of calculations from GPU memory. */
		err = clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, M * sizeof(*result),
			result, 0, NULL, NULL);
		/* At this point you will get the result of SGEMV placed in 'result' array. */
		puts("");
		printResult("clblasSgemvEx result");
	}
	/* Release OpenCL events. */
	clReleaseEvent(event);
	/* Release OpenCL memory objects. */
	clReleaseMemObject(bufY);
	clReleaseMemObject(bufX);
	clReleaseMemObject(bufA);
	/* Finalize work with clblas. */
	clblasTeardown();
	/* Release OpenCL working objects. */
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);
	return ret;
}
