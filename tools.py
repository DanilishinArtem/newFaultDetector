import torch
import numpy as np
import struct

def float_bitflip(number: float, k: int) -> float:
    b = bytearray(struct.pack("f", number))
    byte_idx = k // 8
    bit_idx = k % 8
    b[byte_idx] ^= (1 << bit_idx)
    return struct.unpack("f", bytes(b))[0]

def float16_bitflip(number: float, k: int) -> np.float16:
    fp16 = np.array(number, dtype="float16")
    bits = np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0]
    bits ^= (1 << k)
    flipped = np.frombuffer(np.uint16(bits).tobytes(), dtype="float16")[0]
    return flipped

def bfloat16_bitflip(number: float, k: int) -> np.float16:
    float32_bytes = struct.pack("f", number)
    upper_2bytes = float32_bytes[2:]  # [2], [3] → старшие биты
    bits = int.from_bytes(upper_2bytes, byteorder='little')
    bits ^= (1 << k)
    flipped_bytes = float32_bytes[:2] + bits.to_bytes(2, byteorder='little')
    flipped_float = struct.unpack("f", flipped_bytes)[0]
    return torch.tensor(flipped_float, dtype=torch.bfloat16)

def bitflip_fp32(input: torch.tensor, k: int):
    rand_ind = np.random.randint(0, input.numel())
    flat = input.flatten()
    original = flat[rand_ind]
    flipped = float_bitflip(original, 31 - k)
    print("[INFO] Biflip: original={}, flipped={}".format(original, flipped))
    flat[rand_ind] = flipped

def bitflip_fp16(input: torch.Tensor, k: int):
    rand_ind = np.random.randint(0, input.numel())
    flat = input.flatten()
    original = flat[rand_ind].item()  # преобразуем в float для numpy
    flipped = float16_bitflip(original, 15 - k)  # numpy.float16
    flipped_torch = torch.tensor(flipped, dtype=torch.float16, device=input.device)  # конвертируем
    print(f"[INFO] Bitflip: original={original}, flipped={flipped_torch.item()}")
    flat[rand_ind] = flipped_torch

def bitflip_bf16(input: torch.Tensor, k: int):
    rand_ind = np.random.randint(0, input.numel())
    flat = input.flatten()
    original = flat[rand_ind].item()
    flipped = bfloat16_bitflip(original, 15 - k)
    print(f"[INFO] Bitflip: original={original}, flipped={flipped.item()}")
    flat[rand_ind] = flipped

def get_random_tensor(m: int, n: int, k: int, dtype: torch.dtype):
    mean = 0.0
    std = 1.0
    if k != -1:
        tensor = torch.randn(m, n, k).abs().log().to(dtype) * std + mean
    else:
        tensor = torch.randn(m, n).abs().log().to(dtype) * std + mean
    return tensor

def get_fault_tensor(tensor, bit: int):
    golden = tensor.clone()
    if tensor.dtype == torch.float:
        assert bit < 31, "[ERROR] For float32 bit parameter should be less then 31"
        bitflip_fp32(tensor, bit)
        return golden, tensor
    elif tensor.dtype == torch.float16:
        assert bit < 15, "[ERROR] For float16 bit parameter should be less then 15"
        bitflip_fp16(tensor, bit)
        return golden, tensor
    elif tensor.dtype == torch.bfloat16:
        assert bit < 15, "[ERROR] For bfloat16 bit parameter should be less then 15"
        bitflip_bf16(tensor, bit)
        return golden, tensor
    else:
        raise NotImplementedError

def get_fault_matmul_first_tensor(first_tensor, second_tensor, bit: int):
    golden = torch.matmul(first_tensor, second_tensor)
    if first_tensor.dtype == torch.float:
        assert bit < 31, "[ERROR] For float32 bit parameter should be less then 31"
        bitflip_fp32(first_tensor, bit)
        return golden, torch.matmul(first_tensor, second_tensor)
    elif first_tensor.dtype == torch.float16:
        assert bit < 15, "[ERROR] For float16 bit parameter should be less then 15"
        bitflip_fp16(first_tensor, bit)
        return golden, torch.matmul(first_tensor, second_tensor)
    elif first_tensor.dtype == torch.bfloat16:
        assert bit < 15, "[ERROR] For bfloat16 bit parameter should be less then 15"
        bitflip_bf16(first_tensor, bit)
        return golden, torch.matmul(first_tensor, second_tensor)
    else:
        raise NotImplementedError

def get_fault_matmul_second_tensor(first_tensor, second_tensor, bit: int):
    golden = torch.matmul(first_tensor, second_tensor)
    if first_tensor.dtype == torch.float:
        assert bit < 31, "[ERROR] For float32 bit parameter should be less then 31"
        bitflip_fp32(second_tensor, bit)
        return golden, torch.matmul(first_tensor, second_tensor)
    elif first_tensor.dtype == torch.float16:
        assert bit < 15, "[ERROR] For float16 bit parameter should be less then 15"
        bitflip_fp16(second_tensor, bit)
        return golden, torch.matmul(first_tensor, second_tensor)
    elif first_tensor.dtype == torch.bfloat16:
        assert bit < 15, "[ERROR] For bfloat16 bit parameter should be less then 15"
        bitflip_bf16(second_tensor, bit)
        return golden, torch.matmul(first_tensor, second_tensor)
    else:
        raise NotImplementedError

def test_functions():
    m, n, k = 100, 50, 100
    dtype = torch.float16
    bit = 1

    print("[INFO] Test get_fault_tensor")
    _, _ = get_fault_tensor(m, n, k, dtype, bit)

    print("[INFO] Test get_fault_matmul_first_tensor")
    _, _ = get_fault_matmul_first_tensor(m, n, k, dtype, bit)

    print("[INFO] Terst get_fault_matmul_second_tensor")
    _, _ = get_fault_matmul_second_tensor(m, n, k, dtype, bit)

    print("[INFO] All tests are passed")


# if __name__ == "__main__":
#     test_functions()
