import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import struct
from typing import Optional
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Fault:
    def __init__(self, fault_step: list, fault_bit: int):
        self.fault_step = fault_step
        self.current_step_forward = 0
        self.current_step_backward = 0
        self.fault_bit = fault_bit

    def float_bitflip(self, number: float, k: int) -> float:
        b = bytearray(struct.pack("f", number))
        byte_idx = k // 8
        bit_idx = k % 8
        b[byte_idx] ^= (1 << bit_idx)
        return struct.unpack("f", bytes(b))[0]

    def bitflip_fp32(self, input: torch.tensor, k: int):
        rand_ind = np.random.randint(0, input.numel())
        flat = input.flatten()
        original = flat[rand_ind]
        flipped = self.float_bitflip(original, 31 - k)
        print("[INFO] Biflip: original={}, flipped={}".format(original, flipped))
        flat[rand_ind] = flipped

    def forward_hook(self, module, input, output):
        self.current_step_forward += 1
        if self.current_step_forward in self.fault_step:
            self.bitflip_fp32(input=input[0].data, k=self.fault_bit)

    def backward_hook(self, module, grad_input, grad_output):
        self.current_step_backward += 1
        if self.current_step_backward in self.fault_step:
            self.bitflip_fp32(input=grad_output[0].data, k=self.fault_bit)

class SafeTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, shift=None, eps=1e-6):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        # Автоматический подбор shift
        min_val = data.min().item()
        auto_shift = max(eps, -min_val + eps)
        shift = auto_shift if shift is None else shift

        # Преобразование: log(x + shift)
        log_data = (data + shift).log()
        instance = log_data.as_subclass(cls)
        instance._shift = shift
        instance._eps = eps
        return instance

    def restore(self):
        return self.exp() - self._shift

    def __repr__(self):
        shift = getattr(self, "_shift", 0.0)
        return f"MicroScaledTensor(log(x + {shift:.2e})): {super().__repr__()}"

    def clone(self, *args, **kwargs):
        restored = self.restore().clone(*args, **kwargs)
        return SafeTensor(restored, shift=self._shift, eps=self._eps)

    def to(self, *args, **kwargs):
        restored = self.restore().to(*args, **kwargs)
        return SafeTensor(restored, shift=self._shift, eps=self._eps)
    

class SafeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

        # SafeTensor-представление (для хранения)
        self.safe_weight: Optional[SafeTensor] = None
        self.safe_bias: Optional[SafeTensor] = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Используем обычные веса в прямом проходе
        return F.linear(input, self.weight, self.bias)

    def update_safe_view(self):
        self.safe_weight = SafeTensor(self.weight.data)
        if self.bias is not None:
            self.safe_bias = SafeTensor(self.bias.data)

    def __repr__(self):
        self.update_safe_view()
        return f"SafeLinear(\n  weight={self.safe_weight},\n  bias={self.safe_bias}\n)"
    

class SafeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = SafeLinear(input_dim, hidden_dim)
        self.fc2 = SafeLinear(hidden_dim, output_dim)

    def forward(self, x: SafeTensor) -> torch.Tensor:
        # Восстанавливаем данные из SafeTensor перед передачей в слои
        x_real = x.restore()  # (B, D)
        x = F.relu(self.fc1(x_real))  # real → relu
        x = self.fc2(x)  # финальный real output
        return x
    

def train(model, dataloader, writer, epochs=10, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            # Прямой проход
            optimizer.zero_grad()
            safe_inputs = SafeTensor(inputs)
            outputs = model(safe_inputs)
            loss = criterion(outputs, targets)
            
            # Обратный проход
            loss.backward()
            # Шаг оптимизации
            optimizer.step()
            total_loss += loss.item()
            
        writer.add_scalar("Loss/train", total_loss, epoch)
        print(f"Эпоха {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


def get_synthetic_data(n_samples=1024, input_dim=784, num_classes=10):
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(X, y)


if __name__ == "__main__":
    
    writer = SummaryWriter("./tensorboard")
    dataset = get_synthetic_data()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = SafeMLP(input_dim=784, hidden_dim=128, output_dim=10)

    fault = Fault(fault_step=[50], fault_bit=1)
    forward_hook = fault.forward_hook
    # backward_hook = fault.backward_hook
    model.fc1.register_forward_hook(forward_hook)
    # model.fc1.register_backward_hook(backward_hook)

    # Обучение с защитой от битфлипов
    train(model, dataloader, writer, epochs=100)
    writer.close()