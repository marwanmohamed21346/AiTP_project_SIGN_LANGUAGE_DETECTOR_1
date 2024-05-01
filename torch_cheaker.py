# cheaker 1
import torch

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)