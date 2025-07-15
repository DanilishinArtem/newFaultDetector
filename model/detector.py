import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch
import numpy as np
import struct



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

def visualize_anomalies(tensor, mask, title):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.hist(tensor.cpu().flatten().numpy(), bins=50, alpha=0.7)
    plt.title(f"Распределение {title}")
    
    plt.subplot(122)
    anomaly_indices = torch.nonzero(mask).cpu().numpy()
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices[:, 0], anomaly_indices[:, 1], alpha=0.5)
    plt.title(f"Аномалии ({len(anomaly_indices)} точек)")
    
    plt.tight_layout()
    print(f"[INFO] saving ./{title}_anomalies.png")
    plt.savefig(f"./{title}_anomalies.png")
    plt.close()

class RobustTensor:
    def __init__(self, data, shift=None, eps=1e-12, name=None):
        """
        data: исходный тензор
        shift: сдвиг для логарифмического преобразования
        eps: малая константа для численной стабильности
        name: имя тензора для удобства отладки
        """
        self.original = data.detach().clone()
        self.name = name or "RobustTensor"
        
        # Вычисление сдвига
        with torch.no_grad():
            if shift is None:
                min_val = data.min().item()
                abs_data = data.abs()
                min_abs_nonzero = abs_data[abs_data > 0].min().item() if (abs_data > 0).any() else 1.0
                self.shift = abs(min_val) + min_abs_nonzero + eps
            else:
                self.shift = shift
                
        # Логарифмическое представление
        self.log_repr = torch.log2(data.abs() + self.shift + eps)
        self.sign = torch.sign(data)
        
        # Статистика для детектирования
        self.last_check_diff = 0
        self.error_count = 0
        
    def check_integrity(self, current_data, threshold=0.1):
        """
        Проверка целостности данных
        Возвращает маску с подозрительными элементами
        """
        with torch.no_grad():
            # Вычисляем ожидаемое лог-представление
            expected_log = torch.log2(current_data.abs() + self.shift)
            
            # Сравниваем с оригинальным лог-представлением
            diff = torch.abs(self.log_repr - expected_log)
            relative_diff = diff / (self.log_repr.abs() + 1e-12)
            
            # Создаем маску подозрительных элементов
            anomaly_mask = (diff > threshold) | (relative_diff > 0.01)
            
            # Обновляем статистику
            self.last_check_diff = diff.mean().item()
            if anomaly_mask.any():
                self.error_count += 1
                
            return anomaly_mask
        
    def correct_errors(self, current_data, threshold=0.1):
        """
        Автокоррекция обнаруженных ошибок
        """
        anomaly_mask = self.check_integrity(current_data, threshold)
        
        with torch.no_grad():
            # Восстанавливаем данные из лог-представления
            restored = self.sign * (2 ** self.log_repr - self.shift)
            
            # Применяем коррекцию только к поврежденным элементам
            corrected_data = current_data.clone()
            corrected_data[anomaly_mask] = restored[anomaly_mask]
            
            # Обновляем текущее состояние
            if anomaly_mask.any():
                self.original.copy_(corrected_data)
                
            return corrected_data, anomaly_mask
        
    def update(self, new_data):
        """Обновление состояния тензора"""
        with torch.no_grad():
            self.original.copy_(new_data)
            abs_data = new_data.abs() + self.shift
            self.log_repr = torch.log2(abs_data)
            self.sign = torch.sign(new_data)

class BitflipMonitor:
    def __init__(self, model, check_interval=10, threshold=0.1):
        """
        model: модель PyTorch для мониторинга
        check_interval: периодичность проверок (в шагах)
        threshold: порог детектирования аномалий
        """
        self.model = model
        self.check_interval = check_interval
        self.threshold = threshold
        self.step_count = 0
        self.robust_params = {}
        
        # Инициализация мониторинга параметров
        for name, param in model.named_parameters():
            self.robust_params[name] = RobustTensor(param.data.clone(), name=name)
        
    def check_weights(self):
        """Проверка целостности весов"""
        anomalies = {}
        for name, param in self.model.named_parameters():
            robust_tensor = self.robust_params[name]
            anomaly_mask = robust_tensor.check_integrity(param.data, self.threshold)
            
            if anomaly_mask.any():
                anomalies[name] = {
                    'mask': anomaly_mask,
                    'diff': robust_tensor.last_check_diff,
                    'count': robust_tensor.error_count
                }
        return anomalies
    
    def correct_weights(self):
        """Коррекция весов на основе лог-представления"""
        corrections = {}
        for name, param in self.model.named_parameters():
            robust_tensor = self.robust_params[name]
            corrected_data, anomaly_mask = robust_tensor.correct_errors(param.data, self.threshold)
            
            if anomaly_mask.any():
                param.data.copy_(corrected_data)
                corrections[name] = {
                    'num_corrected': anomaly_mask.sum().item(),
                    'max_diff': robust_tensor.last_check_diff
                }
        return corrections
    
    def check_gradients(self):
        """Проверка целостности градиентов"""
        anomalies = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            # Создаем временный RobustTensor для градиента
            grad_tensor = RobustTensor(param.grad.clone(), name=f"{name}.grad")
            anomaly_mask = grad_tensor.check_integrity(param.grad, self.threshold)
            
            if anomaly_mask.any():
                anomalies[name] = {
                    'mask': anomaly_mask,
                    'diff': grad_tensor.last_check_diff
                }
        return anomalies
    
    def on_train_step(self, optimizer):
        """Вызывать после каждого шага обучения"""
        self.step_count += 1
        
        # Обновление состояний после оптимизации
        if self.step_count % self.check_interval == 0:
            for name, param in self.model.named_parameters():
                self.robust_params[name].update(param.data)
        
        # Периодическая проверка
        if self.step_count % self.check_interval == 0:
            weight_anomalies = self.check_weights()
            grad_anomalies = self.check_gradients()
            
            # Автокоррекция весов
            if weight_anomalies:
                corrections = self.correct_weights()
                return {
                    'weight_anomalies': weight_anomalies,
                    'grad_anomalies': grad_anomalies,
                    'corrections': corrections
                }
        return {}
    

class SafeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train(model, dataloader, epochs=10, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Инициализация мониторинга
    monitor = BitflipMonitor(model, check_interval=50)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            # Прямой проход
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Обратный проход
            loss.backward()
            # Шаг оптимизации
            optimizer.step()
            total_loss += loss.item()

            # Проверка на битфлипы
            monitor_result = monitor.on_train_step(optimizer)
            
            if monitor_result:
                print(f"\nШаг {i}: Обнаружены аномалии!")

                for name, info in monitor_result['weight_anomalies'].items():
                    param = next(p for n, p in model.named_parameters() if n == name)
                    visualize_anomalies(param.data, info['mask'], f"{i} Веса {name}")

                for name, info in monitor_result['grad_anomalies'].items():
                    param = next(p for n, p in model.named_parameters() if n == name)
                    visualize_anomalies(param.data, info['mask'], f"{i} Градиенты {name}")

                for name, info in monitor_result.get('weight_anomalies', {}).items():
                    print(f" - Веса {name}: {info['diff']:.4f} расхождение")
                
                for name, info in monitor_result.get('grad_anomalies', {}).items():
                    print(f" - Градиенты {name}: {info['diff']:.4f} расхождение")
                
                if 'corrections' in monitor_result:
                    for name, info in monitor_result['corrections'].items():
                        print(f" - Корректировка {name}: исправлено {info['num_corrected']} элементов")
        
        print(f"Эпоха {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


def get_synthetic_data(n_samples=1024, input_dim=784, num_classes=10):
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(X, y)


if __name__ == "__main__":
    # Создание модели и данных
    model = SafeMLP(784, 128, 10)
    fault = Fault(fault_step=[50, 51], fault_bit=1)
    # forward_hook = fault.forward_hook
    backward_hook = fault.backward_hook
    # model.fc1.register_forward_hook(forward_hook)
    model.fc1.register_forward_hook(backward_hook)
    dataset = get_synthetic_data()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Обучение с защитой от битфлипов
    train(model, dataloader, epochs=100)
