#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for memory access
__global__ void memory_latency_test(int *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int value = data[idx];
        // Perform some operations to ensure memory access
        data[idx] = value + 1;
    }
}

int main() {
    int size = 1; // 1K elements
    int *h_data = (int*)malloc(size * sizeof(int));
    int *d_data;
    cudaMalloc(&d_data, size * sizeof(int));

    // Initialize host data
    for (int i = 0; i < size; ++i) {
        h_data[i] = i;
    }

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Measure memory latency
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    memory_latency_test<<<(size + 255) / 256, 256>>>(d_data, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Memory latency: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}