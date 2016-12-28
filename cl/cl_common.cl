
kernel void caffe_copy(global const float *X, global float *Y)
{
	int gid = get_global_id(0);
	Y[gid] = X[gid];
}

kernel void im2col(const int n, global float* data_im, const int off_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    global float* data_col)
{
	//int ls = get_local_size(0);
	//int ng = get_num_groups(0);
	//int gi = get_group_id(0);
	//int li = get_local_id(0);
	//printf("%d; %d; %d \n", a, b, a * b);
	data_im += off_im;
	// get_group_id(0)    blockIdx.x
	// get_local_size(0)  blockDim.x
	// get_local_id(0)    threadIdx.x
	// get_num_groups(0)  gridDim.x  
	//for ( int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index +=blockDim.x * gridDim.x) 
  	// for (int index = gi * ls + li;  index < n;  index += ls * ng) {
	int index = get_global_id(0);
    		const int h_index = index / width_col;
    		const int h_col = h_index % height_col;
    		const int w_col = index % width_col;
    		const int c_im = h_index / height_col;
    		const int c_col = c_im * kernel_h * kernel_w;
    		const int h_offset = h_col * stride_h - pad_h;
    		const int w_offset = w_col * stride_w - pad_w;
    		global float* data_col_ptr = data_col;
    		data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    		global const float* data_im_ptr = data_im;
    		data_im_ptr += (c_im * height + h_offset) * width + w_offset;
			
    		for (int i = 0; i < kernel_h; ++i) {
      			for (int j = 0; j < kernel_w; ++j) {
        			int h_im = h_offset + i * dilation_h;
        			int w_im = w_offset + j * dilation_w;
        			*data_col_ptr =
            				(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            				data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        			data_col_ptr += height_col * width_col;
      			}
    		}
  	//}
}


kernel void im2col_nd(const int num_axes, const int n, global float* data_im,const int off_im,
    global const int* im_shape, global const int* col_shape,
    global const int* kernel_shape, global const int* pad, global const int* stride,
    global const int* dilation, global float* data_col) 
{
	data_im += off_im;
	int d_temp[10];
	int d_iter[10];

	local int shared_dilation[10];
	local int shared_kernel_shape[10];
	local int shared_pad[10];
	local int shared_stride[10];
	local int shared_col_shape[11];
	local int shared_im_shape[11];
	int threadIdx_x = get_local_id(0);

  if (threadIdx_x < num_axes) {
    shared_dilation[threadIdx_x] = dilation[threadIdx_x];
    shared_kernel_shape[threadIdx_x] = kernel_shape[threadIdx_x];
    shared_pad[threadIdx_x] = pad[threadIdx_x];
    shared_stride[threadIdx_x] = stride[threadIdx_x];
  }
  if (threadIdx_x < num_axes + 1) {
    shared_col_shape[threadIdx_x] = col_shape[threadIdx_x];
    shared_im_shape[threadIdx_x] = im_shape[threadIdx_x];
  }
  //__syncthreads();
	barrier(CLK_GLOBAL_MEM_FENCE);
  int i;
  //CUDA_KERNEL_LOOP(index, n)
  for (int index = get_group_id(0) * get_local_size(0) + get_local_id(0);  index < n;  index += get_local_size(0) * get_num_groups(0))
  {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_in = index;
    int channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % shared_col_shape[i + 1];
      channel_in /= shared_col_shape[i + 1];
      channel_out *= shared_kernel_shape[i];
    }
    channel_out *= channel_in;
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= shared_col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      channel_in *= shared_im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= shared_col_shape[i + 1];
      d_iter[i] = 0;
    }
    global float* data_col_ptr = data_col + channel_out;
    global const float* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        if (!in_range) { break; }
      }
      if (in_range) {
        int data_im_offset = d_iter[0] * shared_dilation[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= shared_im_shape[i + 1];
          data_im_offset += d_iter[i] * shared_dilation[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int d_max = shared_kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}


kernel void col2im(const int n, const global float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    global float* data_im,const int off_im) 
{
	//int ls = get_local_size(0);
	//int ng = get_num_groups(0);
	//int gi = get_group_id(0);
	//int li = get_local_id(0);
    data_im += off_im;

    //CUDA_KERNEL_LOOP(index, n) 
    //for (int index = gi * ls + li;  index < n;  index += ls * ng)    
    //for (int index = gi * ls + li;  index < n;  index += ls * ng) {
    int index = get_global_id(0);
    float val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  //}
}


kernel void col2im_nd(const int num_axes,const int n, global const float* data_col,
    global const int* im_shape, global const int* col_shape,
    global const int* kernel_shape, global const int* pad, global const int* stride,
    global const int* dilation, global float* data_im,const int off_im) 
{

  data_im += off_im;
  int d_im[10];  // NOLINT(runtime/arrays)
  int d_col_iter[10];  // NOLINT(runtime/arrays)
  int d_col_start[10];  // NOLINT(runtime/arrays)
  int d_col_end[10];  // NOLINT(runtime/arrays)

  local int shared_dilation[10];
  local int shared_kernel_shape[10];
  local int shared_pad[10];
  local int shared_stride[10];
  local int shared_col_shape[11];
  local int shared_im_shape[11];

  int threadIdx_x = get_local_id(0);
  if (threadIdx_x < num_axes) {
    shared_dilation[threadIdx_x] = dilation[threadIdx_x];
    shared_kernel_shape[threadIdx_x] = kernel_shape[threadIdx_x];
    shared_pad[threadIdx_x] = pad[threadIdx_x];
    shared_stride[threadIdx_x] = stride[threadIdx_x];
  }
  if (threadIdx_x < num_axes + 1) {
    shared_col_shape[threadIdx_x] = col_shape[threadIdx_x];
    shared_im_shape[threadIdx_x] = im_shape[threadIdx_x];
  }

  //__syncthreads();

   barrier(CLK_GLOBAL_MEM_FENCE);
  //int i;
  //CUDA_KERNEL_LOOP(index, n)
  for (int index = get_group_id(0) * get_local_size(0) + get_local_id(0);  index < n;  index += get_local_size(0) * get_num_groups(0))
  {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int c_im = index;
    // Calculate d_im (image dimensions).
    for (int i = num_axes - 1; i >= 0; --i) {
      d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];
      c_im /= shared_im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int i = 0; i < num_axes; ++i) {
      const int kernel_extent =
          shared_dilation[i] * (shared_kernel_shape[i] - 1) + 1;
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_extent) ? 0 :
          (d_im[i] - kernel_extent) / shared_stride[i] + 1;
      d_col_end[i] =
          min(d_im[i] / shared_stride[i] + 1, shared_col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    float val = 0;
    bool incremented = true;
    bool skip = false;
    do {
      // Compute the final offset.
      int final_offset = 0;
      int kernel_shape_prod = 1;
      int kernel_index;
      for (int i = num_axes - 1; i >= 0; --i) {
        kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];
        if (kernel_index % shared_dilation[i]) {
          skip = true;
          break;
        } else {
          kernel_index /= shared_dilation[i];
          final_offset += kernel_index * kernel_shape_prod;
          kernel_shape_prod *= shared_kernel_shape[i];
        }
      }
      if (!skip) {
        final_offset += kernel_shape_prod * c_im;
        for (int i = 0; i < num_axes; ++i) {
          final_offset *= shared_col_shape[i + 1];
          final_offset += d_col_iter[i];
        }
        val += data_col[final_offset];
      }
      skip = false;
      incremented = false;
      for (int i = num_axes - 1; i >= 0; --i) {
        const int d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    data_im[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}
