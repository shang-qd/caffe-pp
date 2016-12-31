
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
