
kernel void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}
