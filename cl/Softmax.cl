kernel void channel_max(const int num, const int channels,
    const int spatial_dim, global const float* data, global float* out) {
	int index = get_global_id(0);
	if (index >= num * spatial_dim) { return; }

    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
}

kernel void channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, global const float* channel_max, global float* data) {
  	int index = get_global_id(0);
	if (index >= count) { return; }
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  
}

kernel void channel_exp(const int count, global const float* data, global float* out) {
	int index = get_global_id(0);
	if (index >= count) { return; }
    out[index] = exp(data[index]);  
}

kernel void channel_sum(const int num, const int channels,
    const int spatial_dim, global const float* data, global float* channel_sum) {
	int index = get_global_id(0);
	if (index >= num * spatial_dim) { return; }
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
}

kernel void channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, global const float* channel_sum, global float* data) {
	int index = get_global_id(0);
	if (index >= count) { return; }
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];

}

kernel void channel_dot(const int num, const int channels,
    const int spatial_dim, global const float* data_1, global const float* data_2,
    global float* channel_dot) {
  	int index = get_global_id(0);
	if (index >= num * spatial_dim) { return; }
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
}
