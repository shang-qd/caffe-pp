
kernel void ReLU_Forward(const int n, global const float *bottom_data,
			global float *top_data,
			const float negative_slope)
{
	int gid = get_global_id(0);
	if (gid >= n) { return; }
	if (bottom_data[gid] > 0)
	{
		top_data[gid] = bottom_data[gid];
	}
	else
	{
		top_data[gid] = negative_slope * bottom_data[gid];
	}
}

kernel void ReLU_Backward(const int n, global const float *bottom_data,
			global const float *top_diff,
			global float *bottom_diff,
			const float negative_slope)
{
	
	int gid = get_global_id(0);
	if (gid >= n) { return; }
	if (bottom_data[gid] > 0)
	{
		bottom_diff[gid] = top_diff[gid];
	}
	else
	{
		bottom_diff[gid] = negative_slope * top_diff[gid];
	}

}

