import torch
import torch.nn as nn

window_len = 8
stride = 1
fair_comparison = True

device = "cuda" if torch.cuda.is_available() else "cpu"

dropout = 0.25
learning_rate = 0.0002
num_epochs = 20
chunk_size = 64
forward_chunk = 8 
forward_chunk_size = 8 # this is smaller due to memory constrains 

loss_fn = nn.MSELoss()
