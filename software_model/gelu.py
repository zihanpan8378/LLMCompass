from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from energy_model.energy_model import EnergyModel
from math import ceil, log2, log
import time
import statistics
import numpy as np
import torch
import pynvml


@torch.compile
def gelu_gpu(input: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(input, approximate="tanh")


# x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
class GeLU(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:])
        self.computational_graph = self.ComputationalGraph(self.M, self.data_type)
        return input

    def roofline_model(self, pcb_module: Device):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        M = self.M
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = (
            total_io_count / min(pcb_module.io_module.bandwidth
            , pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq)
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )
        self.roofline_latency = max(compute_latency, io_latency)
        return self.roofline_latency

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, data_type: DataType):            
            self.M = M
            self.data_type = data_type

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        self.energy_model = EnergyModel(
            process_node=pcb_module.compute_module.process_node,
            memory_node=pcb_module.memory_module.memory_node
        )
        
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        parallelism = (
            pcb_module.compute_module.core_count
            * pcb_module.compute_module.core.vector_unit.vector_width
            * pcb_module.compute_module.core.vector_unit.vector_count
        )
        M = ceil(self.computational_graph.M / parallelism) * parallelism
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = (
            total_io_count / pcb_module.io_module.bandwidth
            + total_io_count
            / pcb_module.compute_module.l2_bandwidth_per_cycle
            / pcb_module.compute_module.clock_freq
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )
        
        energy_consumption = {
            'memory_to_l2_transfer': 0, 
            'l2_to_l1_transfer': 0, 
            'l1_to_l0_transfer': 0, 
            'compute': 0
        }
        energy_consumption['compute'] = self.energy_model.compute(total_flop_count)
        energy_consumption['memory_to_l2_transfer'] = self.energy_model.transfer_memory_l2(total_io_count * 8)
        energy_consumption['l2_to_l1_transfer'] = self.energy_model.transfer_l2_l1(total_io_count * 8)
        energy_consumption['total'] = sum(energy_consumption.values())
        self.energy_consumption = energy_consumption

        return max(compute_latency, io_latency), self.energy_consumption

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []

        # warmup
        for _ in range(3):
            _ = gelu_gpu(input)
            torch.cuda.synchronize()
            
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)    
        
        latencies = []
        total_iterations = 0
        iterations_start = time.time()
        graphics_freq = []
        count = 0.5
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        while True:
            for _ in range(self.iterations):
                start = time.time()
                output = gelu_gpu(input)
                torch.cuda.synchronize()
                end = time.time()
                assert output.shape == input.shape
                latencies.append(end - start)
            total_iterations += self.iterations
            current_time = time.time()
            if ((current_time - iterations_start) >= count):
                graphics_freq.append(pynvml.nvmlDeviceGetClockInfo(device, pynvml.NVML_CLOCK_GRAPHICS))
                count += 0.5
            if ((current_time - iterations_start) >= 3):
                break
        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        pynvml.nvmlShutdown()
        
        median_latency = statistics.median(latencies)
        
        self.latency_on_gpu = median_latency
        return median_latency, (end_energy - start_energy) / total_iterations, statistics.mean(graphics_freq)

    @staticmethod
    def gpu_kernel_launch_overhead():
        import torch

        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = gelu_gpu(a)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        print(latencies)
        return avg_overhead
