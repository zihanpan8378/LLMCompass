from software_model.matmul import Matmul
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
    gpu_overhead = 2.1e-5
    
    device = RTX4090
    
    device_name = "RTX4090"
    print(f"Device: {device_name} {device}")
    
    file_name=f'ae/energy_experiments/matmul_{device_name}_sim.csv'
    if os.path.exists(file_name):
        os.remove(file_name)
        with open(file_name, 'w') as f:
            f.write('')
    
    if args.gpu:
        gpu_kernel_launch_overhead = Matmul.gpu_kernel_launch_overhead()

    K = 12288
    N = K
    titile = f"Performance of Matmul with K={K}, N={N}"
    print(f"{titile}")
    
    for M in range(5, 10):
        M = 2**M
        model = Matmul(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, K]),
            Tensor([K, N]),
        )
        if args.gpu:
            if test_overhead:
                model.gpu_kernel_launch_overhead()
                test_overhead = False
            latency = model.run_on_gpu()
        if args.simgpu:
            result = model.compile_and_simulate(pcb_module=device, compile_mode="heuristic-GPU")
            latency = result[0] + 2.1e-5
            energy = result[1]
            
        tflops = 2 * M * N * K / latency / 1e12
        print(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops, {energy}", flush=True)
        with open(file_name, 'a') as f:
            f.write(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops, {energy}\n")

    M = 8192
    print(f"Performance of Matmul with M={M}, N=K")
    for K in range(5, 10):
        K = 2**K
        N = K
        model = Matmul(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, K]),
            Tensor([K, N]),
        )
        if args.gpu:
            latency = model.run_on_gpu()
        if args.simgpu:
            result = model.compile_and_simulate(pcb_module=device, compile_mode="heuristic-GPU")
            latency = result[0] + 2.1e-5
            energy = result[1]
            
        tflops = 2 * M * N * K / latency / 1e12
        print(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops, {energy}", flush=True)
        with open(file_name, 'a') as f:
            f.write(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops, {energy}\n")