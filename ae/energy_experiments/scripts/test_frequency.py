from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
import argparse
import os
import subprocess
import time
import torch
import pynvml

def set_gpu_frequency(freq):
    try:
        subprocess.run(["sudo", "nvidia-smi", "-ac", f"{5001},{freq}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to set GPU frequency: {e}")

def reset_gpu_frequency():
    try:
        subprocess.run(["sudo", "nvidia-smi", "-rac"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to reset GPU frequency: {e}")

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
        gpu_overhead = 2.1e-5
    elif device_name == "RTX4090":
        system = template_to_system(read_architecture_template("configs/RTX4090.json"))
        gpu_overhead = 2.05e-5
    elif device_name == "RTX6000Ada":
        system = template_to_system(read_architecture_template("configs/RTX6000Ada.json"))
        gpu_overhead = 2.22e-5
    elif device_name == "L4":
        system = template_to_system(read_architecture_template("configs/L4.json"))
        gpu_overhead = 2.22e-5
    
    device = system.device
    
    print(f"Device: {device_name} {device}")
    
    if args.mode == "run":
        args.gpu = True
        file_name = f'ae/energy_experiments/frequency_{device_name}_gpu.csv'
        gpu_kernel_launch_overhead = Matmul.gpu_kernel_launch_overhead()
    elif args.mode == "sim":
        args.simgpu = True
        file_name = f'ae/energy_experiments/frequency_{device_name}_sim.csv'
    else:
        raise ValueError("Invalid mode")
    
    if os.path.exists(file_name):
        os.remove(file_name)
        with open(file_name, 'w') as f:
            f.write('')

    start_freq = 615
    end_freq = 2505
    step = 45
    
    temp_up = True

    K = 12288
    N = K
    M = 2048
    titile = f"Performance of Matmul with K={K}, N={N}, M={M}"
    print(f"{titile}")
    

    for freq in range(start_freq, end_freq + 1, step):
        
        model = Matmul(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, K]),
            Tensor([K, N]),
        )
        
        if not (temp_up):
            target_temp = 67
            reset_gpu_frequency()
            print(f"Heat up GPU to {target_temp}°C")
            input1 = torch.randn(49152, 49152, dtype=torch.float16, device="cuda:0")
            input2 = torch.randn(49152, 49152, dtype=torch.float16, device="cuda:0")
            
            pynvml.nvmlInit()
            device = pynvml.nvmlDeviceGetHandleByIndex(0)
            while True:
                torch.matmul(input1, input2)
                torch.cuda.synchronize()
                current_temp = pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)
                if current_temp >= target_temp:
                    break
            pynvml.nvmlShutdown()
        
        
        print(f"Setting GPU frequency to {freq} MHz")
        set_gpu_frequency(freq)
        
        # time.sleep(1)
        
        latency, latency_average, energy, grephics_freq, temperature, powers = model.run_on_gpu_lim_temp(temp_up)
        energy *= 1e9
        tflops = 2 * M * N * K / latency / 1e12
        power = (energy / 1e12) / latency
        print(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {latency_average*1e3:.4f}ms, {tflops:.4f}Tflops, {power:.2f}W, {energy}pJ, {grephics_freq}MHz, , {temperature}, {powers}", flush=True)
        with open(file_name, 'a') as f:
            f.write(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops, {power:.2f}W, {energy}, {grephics_freq}, {temperature}, {powers}\n")
    
        time.sleep(1)
        print()
        reset_gpu_frequency()
        
        if temp_up:
            time.sleep(5)
    
    print("Resetting GPU frequency to default")
    
    print("\n")