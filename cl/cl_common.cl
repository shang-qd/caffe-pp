
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
