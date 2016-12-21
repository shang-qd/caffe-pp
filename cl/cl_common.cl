
kernel void Caffe_Copy(global const float *X, global float *Y)
{
	int gid = get_global_id(0);
	Y[gid] = X[gid];
}
