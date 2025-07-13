import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tools import *

def plot_tensor(tensor: torch.tensor, name):
    tensor = tensor.float().flatten().cpu().numpy()
    # Рисуем гистограмму
    plt.hist(tensor, bins=50, edgecolor='black')
    plt.title("[Histogram] {}".format(name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./histogram_{}.png".format(name))

def plot_two_tensors(name, first_tensor: torch.tensor, first_tensor_name: str, second_tensor: torch.tensor, second_tensor_name: str):
    first_tensor = first_tensor.float().flatten().cpu().numpy()
    second_tensor = second_tensor.float().flatten().cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(first_tensor, bins=50, alpha=0.3, label=first_tensor_name, color='blue')
    plt.hist(second_tensor, bins=50, alpha=0.3, label=second_tensor_name, color='red')
    plt.legend()
    plt.title("[{}] Histogram: {} vs {}".format(name, first_tensor_name, second_tensor_name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./{}_histogram_{}_vs_{}.png".format(name, first_tensor_name, second_tensor_name))



class Tensor:
    def __init__(self, tensor: torch.tensor):
        self.shift = None
        self.data = self._transformt_to(tensor)
    
    def _transformt_to(self, tensor: torch.tensor):
        # self.shift = tensor.min().abs().item() + tensor.abs().min().item()
        self.shift = tensor.min().abs().item() + tensor.abs().where(tensor.abs() > 0, torch.tensor([9999.0])).min().item()
        print("[INFO] shift_value = {}".format(self.shift))
        return (tensor + self.shift).log2()

    def get_reg_tensor(self):
        return pow(2, self.data) - self.shift

    def matmul(self, other):
        return Tensor(torch.matmul(pow(2, self.data) - self.shift, pow(2, other.data) - other.shift))

    def check_fault(self):
        pass


class Timer:
    def __init__(self):
        self._start = 0
    def start(self):
        self._start = time.time()
    def end(self, name):
        diff = time.time() - self._start
        print("[INFO] Time of {} calculation: {}".format(name, diff))
        return diff
        

if __name__ == "__main__":
    clock = Timer()
    m, n, k = 8, 8096, 8096
    bit = 7
    dtype = torch.bfloat16
    coeff = 1e-5
    
    first_tensor = (torch.randn(m, n) * coeff).abs().log2().to(dtype)

    # golden_tensor = first_tensor.clone()
    # bitflip_bf16(first_tensor, bit)

    # plot_two_tensors("test", golden_tensor, "golden tensor", first_tensor, "flipped tensor")



    second_tensor = torch.rand(n, k).to(dtype)
    safe_first_tensor = Tensor(first_tensor)
    safe_second_tensor = Tensor(second_tensor)
    safe_golden = safe_first_tensor.matmul(safe_second_tensor)

    plot_tensor(first_tensor.data, "first_tensor_no_fault")

    bitflip_bf16(first_tensor, bit)
    plot_tensor(first_tensor.data, "first_tensor_fault")
    clock.start()
    golden = torch.matmul(first_tensor, second_tensor)
    vanilla_time = clock.end("vanilla matmul")

    bitflip_bf16(safe_first_tensor.data, bit)
    clock.start()
    safe_matmul = safe_first_tensor.matmul(safe_second_tensor)
    safe_time = clock.end("safe matmul")

    print("[INFO] rate = {}".format(safe_time / vanilla_time))
    plot_two_tensors("SOURCE", golden, "golden", safe_matmul.get_reg_tensor(), "safe_matmul")
    plot_two_tensors("LOG2", safe_golden.data, "safe_golden", safe_matmul.data, "safe_matmul")