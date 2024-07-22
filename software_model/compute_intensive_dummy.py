from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from energy_model.energy_model import EnergyModel
from math import ceil, log2, floor
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy
import pynvml

class ComputeIntensiveKernel(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        
    def __call__(self, size: int):
        self.size = size
        
    def run_on_gpu(self):
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        matrix_size = self.size
        
        input_1 = torch.randn(
            1024,
            matrix_size,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        input_2 = torch.randn(
            matrix_size,
            1024,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        
        _ = input_1.sum()
        _ = input_2.sum()
        torch.cuda.synchronize()
        
        num_flop = 2 * (1024 * matrix_size * 1024) * self.iterations
        
        print("")
        print(f"Input shape: {input_1.shape} * {input_2.shape}")
        print(f"Num flop: {2 * (1024 * matrix_size * 1024) * self.iterations}")
        #print(f"Num flop: {2 * (matrix_size ** 3) - (matrix_size ** 2)}")
        
        latencies = []
        iterations = 0
        iterations_start = time.time()
        graphics_freq = []
        utilizations = []
        count = 0.5
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        while True:
            start = time.time()
            for i in range(self.iterations):
            #for i in range(10000):
                #_ = torch.dot(input_1.view(-1), input_2.view(-1))
                _ = torch.matmul(input_1, input_2)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
            iterations += 1
            current_time = time.time()
            if ((current_time - iterations_start) >= count):
                utilizations.append(pynvml.nvmlDeviceGetUtilizationRates(device))
                graphics_freq.append(pynvml.nvmlDeviceGetClockInfo(device, pynvml.NVML_CLOCK_GRAPHICS))
                count += 0.5
            if ((current_time - iterations_start) >= 3):
                break
        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        
        pynvml.nvmlShutdown()
        
        median_latency = statistics.median(latencies)
        mean_graphics_freq = statistics.mean(graphics_freq)
        
        gpu_utilizations = [utilizations[i].gpu for i in range(len(utilizations))]
        memory_utilizations = [utilizations[i].memory for i in range(len(utilizations))]
        mean_gpu_utilization = statistics.mean(gpu_utilizations)
        mean_memory_utilization = statistics.mean(memory_utilizations)
        
        
        return median_latency, (end_energy - start_energy) / iterations, mean_graphics_freq, mean_gpu_utilization, num_flop