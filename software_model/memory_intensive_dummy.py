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

class MemoryIntensiveKernel(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        
    def __call__(self, M: int):
        self.M = M
        
    def run_on_gpu(self):
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        size = self.M * 256 * 512
        
        input1 = torch.randn(
            size,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        input2 = torch.empty_like(input1, device="cuda:0")
        
        latencies = []
        iterations = 0
        iterations_start = time.time()
        graphics_freq = []
        count = 0.5
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        while True:
            start = time.time()
            # input2.copy_(input1)
            input2 = torch.zeros_like(input2, device="cuda:0")
            # input2 = torch.ones_like(input2, device="cuda:0")
            end = time.time()
            latencies.append(end - start)
            iterations += 1
            current_time = time.time()
            if ((current_time - iterations_start) >= count):
                graphics_freq.append(pynvml.nvmlDeviceGetClockInfo(device, pynvml.NVML_CLOCK_GRAPHICS))
                count += 0.5
            if ((current_time - iterations_start) >= 10):
                break
        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        
        pynvml.nvmlShutdown()
        
        median_latency = statistics.median(latencies)
        median_graphics_freq = statistics.median(graphics_freq)
        
        return median_latency, (end_energy - start_energy) / iterations, median_graphics_freq