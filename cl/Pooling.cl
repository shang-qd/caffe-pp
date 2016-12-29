kernel void MaxPoolForward(global float* bottom_data, const int num, const int channels,
    	const int height, const int width, const int pooled_height,
    	const int pooled_width, const int kernel_h, const int kernel_w,
    	const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    	global float* top_data, global int* mask, global float* top_mask) {
	
	int index = get_global_id(0);
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;

	bottom_data += (n * channels + c) * height * width;
	for (int h = hstart; h < hend; ++h) {
      	for (int w = wstart; w < wend; ++w) {
        	if (bottom_data[h * width + w] > maxval) {
          		maxidx = h * width + w;
          		maxval = bottom_data[maxidx];
        	}
      	}
    }
    top_data[index] = maxval;
    if (mask) {
		mask[index] = maxidx;
    } else {
		top_mask[index] = maxidx;
    }
}


kernel void MaxPoolBackward(global const float* top_diff, global const int* mask, 
	global const float* top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, global float *bottom_diff) {
	int index = get_global_id(0);

    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    float gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
	top_diff = top_diff + offset;
	mask = mask + offset;
	top_mask = top_mask + offset;
    if (mask) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    } else {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
}
