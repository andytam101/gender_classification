import torch
from utilities import *

val = torch.load("data/val_data.pt", map_location=device)

val_X = val[:, 1:]
val_y = val[:, 0]

print(cost(val_X, convert_y(val_y)))
