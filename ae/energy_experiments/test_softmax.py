from software_model.softmax import Softmax
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    args = parser.parse_args()

    A100 = device_dict["A100_80GB_fp16"]
    RTX4090 = template_to_system(read_architecture_template("configs/RTX4090.json")).device
    gpu_overhead = 12.1e-6
    
    device = RTX4090
    
    device_name = "RTX4090"
    print(f"Device: {device_name} {device}")
    
    file_name=f'ae/energy_experiments/matmul_{device_name}_sim.csv'
    if os.path.exists(file_name):
        os.remove(file_name)
        with open(file_name, 'w') as f:
            f.write('')

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
            result = model.compile_and_simulate(pcb_module=device)
            latency = result[0] + gpu_overhead
            energy = result[1]
            
        print(f"Performance: {M}, {N}, {M*N/latency/1e9}, {energy}")
        with open(file_name, "a") as f:
            f.write(f"{M}, {N}, {M*N/latency/1e9} {energy}\n")

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
            result = model.compile_and_simulate(pcb_module=device)
            latency = result[0] + gpu_overhead
            energy = result[1]
        
        print(f"Performance: {M}, {N}, {M*N/latency/1e9}, {energy}")
        with open(file_name, "a") as f:
            f.write(f"{M}, {N}, {M*N/latency/1e9} {energy}\n")
