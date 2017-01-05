
kernel void caffe_copy(const int n, global const float *X, global float *Y)
{
	int gid = get_global_id(0);
	if (gid >= n) { return; }
	Y[gid] = X[gid];
}

kernel void caffe_set(const int n, const float alpha, global float *Y)
{
	int gid = get_global_id(0);
	if (gid >= n) { return; }
	Y[gid] = alpha;
}

kernel void caffe_mul(const int n, global const float* a,
    global const float* b, global float* y) 
{
	int index = get_global_id(0);
	if (index >= n) { return; }
    y[index] = a[index] * b[index];
}

kernel void caffe_scal(const int n, const float alpha, global float *x)
{
	int index = get_global_id(0);
	if (index >= n) { return; }
	x[index] = x[index] * alpha;
}

/*
kernel void caffe_asum(const int n, global const float* x, global float* res)
{
	int index = get_global_id(0);
	if (index >= 1) { return; }
	for (int i = 0; i < n; i++)
	{
		res += x[i];
	}
}*/

kernel void caffe_asum(global uint4* input, global uint4* output, local uint4* sdata)
{
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    sdata[tid] = input[stride] + input[stride + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int s = localSize >> 1; s > 0; s >>= 1) {
	if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid == 0) 
	output[bid] = sdata[0];
}