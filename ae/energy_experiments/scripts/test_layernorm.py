from software_model.layernorm import LayerNorm
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="sim", help="sim: simulation, run: run on GPU")
    parser.add_argument("-d", "--device", type=str, help="device to simulate: A100, RTX4090")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "A100":
        system = template_to_system(read_architecture_template("configs/GA100.json"))
        gpu_overhead = 12e-6
    elif device_name == "RTX4090":
        system = template_to_system(read_architecture_template("configs/RTX4090.json"))
        gpu_overhead = 5.96e-5
    elif device_name == "RTX6000Ada":
        system = template_to_system(read_architecture_template("configs/RTX6000Ada.json"))
        gpu_overhead = 6.34e-5
    elif device_name == "L4":
        system = template_to_system(read_architecture_template("configs/L4.json"))
        gpu_overhead = 6.16e-5
    
    device = system.device
    
    print(f"Device: {device_name} {device}")
    
    if args.mode == "run":
        args.gpu = True
        file_name = f'ae/energy_experiments/layernorm_{device_name}_gpu.csv'
        gpu_kernel_launch_overhead = LayerNorm.gpu_kernel_launch_overhead()
    elif args.mode == "sim":
        args.simgpu = True
        file_name = f'ae/energy_experiments/layernorm_{device_name}_sim.csv'
    else:
        raise ValueError("Invalid mode")
    
    if os.path.exists(file_name):
        os.remove(file_name)
        with open(file_name, 'w') as f:
            f.write('')

    test_overhead = True
    M = 2**12

    titile = f"Performance of LayerNorm with M={M}"
    print(f"{titile}")
    
    for N in range(5, 20):
        N = 2**N
        model = LayerNorm(data_type=data_type_dict["fp16"])
        _ = model(Tensor([M, N]))
        if args.gpu:
            if test_overhead:
                overhead = model.gpu_kernel_launch_overhead()
                print(f"Overhead: {overhead*1e3:.4f}ms", flush=True)
                test_overhead = False
                gpu_overhead = overhead
                
            latency, energy, grephics_freq = model.run_on_gpu()
            
            energy *= 1e9
            gelements = M * N / latency / 1e9
            power = (energy / 1e12) / latency
            
            print(f"{M}, {N}, {latency*1e3:.4f}ms, {gelements:.4f}GElements, {power:.2f}W, {energy}pJ, {grephics_freq:.2f}MHz", flush=True)
            with open(file_name, 'a') as f:
                f.write(f"{M}, {N}, {latency*1e3:.4f}ms, {gelements:.4f}GElements, {power:.2f}W, {energy}, {grephics_freq:.2f}\n")
        if args.simgpu:
            result = model.compile_and_simulate(pcb_module=device, compile_mode="heuristic-GPU")
            latency = result[0] + gpu_overhead
            energy = result[1]
            
            gelements = M * N / latency / 1e9
            power = (energy['total'] / 1e12) / latency
            
            print(f"Performance: {M}, {N}, {latency*1e3:.4f}ms, {gelements}GElements, {power:.2f}W, {energy}")
            with open(file_name, "a") as f:
                f.write(f"{M}, {N}, {latency*1e3:.4f}ms, {gelements:.4f}GElements, {power:.2f}W, {energy['total']}, {energy['memory_to_l2_transfer']}, {energy['l2_to_l1_transfer']}, {energy['l1_to_l0_transfer']}, {energy['compute']}\n")

    N = 2**12
    titile = f"Performance of LayerNorm with N={N}"
    print(f"{titile}")
    for M in range(5, 20):
        M = 2**M
        model = LayerNorm(data_type=data_type_dict["fp16"])
        _ = model(Tensor([M, N]))
        if args.gpu:
            latency, energy, grephics_freq = model.run_on_gpu()
            
            energy *= 1e9
            gelements = M * N / latency / 1e9
            power = (energy / 1e12) / latency
            
            print(f"{M}, {N}, {latency*1e3:.4f}ms, {gelements:.4f}GElements, {power:.2f}W, {energy}pJ, {grephics_freq:.2f}MHz", flush=True)
            with open(file_name, 'a') as f:
                f.write(f"{M}, {N}, {latency*1e3:.4f}ms, {gelements:.4f}GElements, {power:.2f}W, {energy}, {grephics_freq:.2f}\n")
        if args.simgpu:
            result = model.compile_and_simulate(pcb_module=device, compile_mode="heuristic-GPU")
            latency = result[0] + gpu_overhead
            energy = result[1]
            
            gelements = M * N / latency / 1e9
            power = (energy['total'] / 1e12) / latency
            
            print(f"Performance: {M}, {N}, {latency*1e3:.4f}ms, {gelements}GElements, {power:.2f}W, {energy}")
            with open(file_name, "a") as f:
                f.write(f"{M}, {N}, {latency*1e3:.4f}ms, {gelements:.4f}GElements, {power:.2f}W, {energy['total']}, {energy['memory_to_l2_transfer']}, {energy['l2_to_l1_transfer']}, {energy['l1_to_l0_transfer']}, {energy['compute']}\n")

    print("\n")
