import torch
from tools import get_fault_tensor

if __name__ == "__main__":
    fault_tensor = get_fault_tensor(m, n, k, dtype=torch.float, bit=1)
    