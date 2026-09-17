import torch

x = torch.rand(5000, 5000, device="cuda")
y = torch.rand(5000, 5000, device="cuda")

z = x @ y

print("Device:", z.device)
print("GPU:", torch.cuda.get_device_name(0))