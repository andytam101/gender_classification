import torch

device = torch.device('cpu')

weights_0 = torch.load("weights_0.pt", map_location=device)
weights_1 = torch.load("weights_1.pt", map_location=device)
weights_2 = torch.load("weights_2.pt", map_location=device)


def cost(x, y):
  output = forward_propagation(x)
  return torch.nn.functional.mse_loss(output, y)

def forward_propagation(x):
  a0 = x
  z1 = a0 @ weights_0
  a1 = torch.sigmoid(z1)
  z2 = a1 @ weights_1
  a2 = torch.sigmoid(z2)
  z3 = a2 @ weights_2
  a3 = torch.sigmoid(z3)

  return a3


def convert_y(y):
  m = y.size()[0]
  compare = torch.tensor([0, 1], device=device)
  return (torch.reshape(y, (m, 1)) == compare).float()