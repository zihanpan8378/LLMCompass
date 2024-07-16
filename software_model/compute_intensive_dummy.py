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
        
    def __call__(self, op_num: int):
        self.op_num = op_num
        
    def run_on_gpu(self):
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        input = torch.randn(
            1,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        
        latencies = []
        iterations = 0
        iterations_start = time.time()
        graphics_freq = []
        count = 0.5
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        while True:
            start = time.time()
            for _ in range(self.op_num):
                input = input + 0.0001
            end = time.time()
            latencies.append(end - start)
            iterations += 1
            current_time = time.time()
            if ((current_time - iterations_start) >= count):
                graphics_freq.append(pynvml.nvmlDeviceGetClockInfo(device, pynvml.NVML_CLOCK_GRAPHICS))
                count += 0.5
            if ((current_time - iterations_start) >= 3):
                break
        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        
        pynvml.nvmlShutdown()
        
        mean_latency = statistics.median(latencies)
        mean_graphics_freq = statistics.mean(graphics_freq)
        
        return mean_latency, (end_energy - start_energy) / iterations, mean_graphics_freq