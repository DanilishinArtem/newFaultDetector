import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
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
    def __new__(cls, data: torch.Tensor, shift: float = None, eps: float = 1e-8):
        # clone and detach
        obj = torch.Tensor._make_subclass(cls, data.clone().detach())
        obj._eps = eps
        obj._shift = shift if shift is not None else obj._compute_shift(data)
        obj._logdata = torch.log2(data.abs() + obj._shift + eps)
        return obj

    def _compute_shift(self, data: torch.Tensor):
        min_val = data.min().item()
        nonzero = data[data != 0].abs()
        return abs(min_val) + (nonzero.min().item() if nonzero.numel() > 0 else 1.0)

    def _inverse_transform(self):
        return torch.exp2(self._logdata) - self._shift

    def data(self):
        return self._inverse_transform()

    # пример перегрузки: repr для печати
    def __repr__(self):
        return f"SafeTensor({self._inverse_transform().__repr__()})"

    def clone(self):
        return SafeTensor(self.data(), self._shift, self._eps)

    def to(self, *args, **kwargs):
        return SafeTensor(self.data().to(*args, **kwargs), self._shift, self._eps)
    

class SafeLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: SafeTensor):
        x_real = x.data()
        out = self.linear(x_real)
        return SafeTensor(out)
    

class SafeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = SafeLinear(input_dim, hidden_dim)
        self.fc2 = SafeLinear(hidden_dim, output_dim)

    def forward(self, x: SafeTensor):
        x = F.relu(self.fc1(x).data())  # real tensor to relu
        return self.fc2(SafeTensor(x)).data()  # финальный real output
    

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
    # forward_hook = fault.forward_hook
    backward_hook = fault.backward_hook
    # model.fc1.register_forward_hook(forward_hook)
    model.fc1.register_forward_hook(backward_hook)

    # Обучение с защитой от битфлипов
    train(model, dataloader, writer, epochs=100)
    writer.close()