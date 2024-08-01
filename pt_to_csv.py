import torch
import pandas as pd


data = torch.load("data.pt")
data_np = data.numpy()
df = pd.DataFrame(data_np)
df.to_csv("data.csv", index=False)
