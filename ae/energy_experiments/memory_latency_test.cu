#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime> 
#include <nvml.h>

// CUDA kernel for memory access
__global__ void memory_latency_test(int *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int value = data[idx];
        // Perform some operations to ensure memory access
        data[idx] = (value * 2) + 1;
    }
}

int main() {
    int size = 1024 * 1024 * 1024 * 1; // 1M elements
    int *h_data = (int*)malloc(size * sizeof(int));
    int *d_data;
    cudaMalloc(&d_data, size * sizeof(int));

    srand(time(0));
    for (int i = 0; i < size; ++i) {
        h_data[i] = rand();
    }

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);

    unsigned long long energy1, energy2;

    // Measure memory latency
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvmlDeviceGetTotalEnergyConsumption(device, &energy1);
    cudaEventRecord(start);
    memory_latency_test<<<(size + 255) / 256, 256>>>(d_data, size);
    cudaEventRecord(stop);
    nvmlDeviceGetTotalEnergyConsumption(device, &energy2);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // long long energyUsage = energy2 - energy1;
    std::cout << "Memory latency: " << milliseconds << " ms" << std::endl;
    std::cout << "Energy usage: " << energy2 << " and " << energy1 << std::endl;

    // Cleanup
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}