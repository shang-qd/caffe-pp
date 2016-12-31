kernel void SoftmaxLossForward(const int nthreads,
          global const float* prob_data, global const float* label, global float* loss,
          const int num, const int dim, const int spatial_dim,
          const int has_ignore_label_, const int ignore_label_,
          global float* counts) {
	int index = get_global_id(0);
	if (index >= nthreads) { return; }

    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int)(label[n * spatial_dim + s]);
  
if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      	loss[index] = -log(fmax(prob_data[n * dim + label_value * spatial_dim + s], 1.001e-12f));
      	counts[index] = 1;
    }
}

kernel void SoftmaxLossBackward(const int nthreads, global const float* top,
          global const float* label, global float* bottom_diff, const int num, const int dim,
          const int spatial_dim, const int has_ignore_label_,
          const int ignore_label_, global float* counts) {
	int index = get_global_id(0);
	if (index >= nthreads) { return; }
  
	const int channels = dim / spatial_dim;

    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int)(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
}

