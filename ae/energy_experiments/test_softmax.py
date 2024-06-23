from software_model.softmax import Softmax
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    args = parser.parse_args()

    A100 = device_dict["A100_80GB_fp16"]
    gpu_overhead = 12e-6

    if args.gpu:
        gpu_kernel_launch_overhead = Softmax.gpu_kernel_launch_overhead()

    print(f"Performance of Softmax")
    M = 2**12
    for N in range(5, 16):
        N = 2**N
        model = Softmax(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, N]),
        )
        if args.gpu:
            latency = model.run_on_gpu()
        if args.simgpu:
            result = model.compile_and_simulate(pcb_module=A100)
            latency = result[0] + gpu_overhead
            energy = result[1]
            file_name = "softmax_A100_sim"
            
        print(f"Performance: {M}, {N}, {M*N/latency/1e9}, {energy}")
        with open(f"ae/energy_experiments/{file_name}_perf.csv", "a") as f:
            f.write(f"{M}, {N}, {M*N/latency/1e9}\n")
            
        with open(f"ae/energy_experiments/{file_name}_energy.csv", "a") as f:
            f.write(f"{M}, {N}, {energy}\n")

    N = 2**12
    for M in range(5, 16):
        M = 2**M
        model = Softmax(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, N]),
        )
        if args.gpu:
            latency = model.run_on_gpu()
        if args.simgpu:
            result = model.compile_and_simulate(pcb_module=A100)
            latency = result[0] + gpu_overhead
            energy = result[1]
        
        print(f"Performance: {M}, {N}, {M*N/latency/1e9}, {energy}")
        with open(f"ae/energy_experiments/{file_name}_perf.csv", "a") as f:
            f.write(f"{M}, {N}, {M*N/latency/1e9}\n")
            
        with open(f"ae/energy_experiments/{file_name}_energy.csv", "a") as f:
            f.write(f"{M}, {N}, {energy}\n")
