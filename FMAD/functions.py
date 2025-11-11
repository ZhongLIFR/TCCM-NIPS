import torch
import torch.nn as nn

import hashlib
import time
import os
import math

def generate_filename():
    salt = os.urandom(16)
    now = str(time.time()).encode()
    hash_digest = hashlib.sha256(salt + now).hexdigest()
    return f"{hash_digest}"


# Sine/cosine encoding of time (Sinusoidal Positional Encoding), used in Transformers
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freq = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freq", freq)

    def forward(self, t):
        # t: [B, 1]
        t = t.view(-1, 1)
        args = t * self.freq  # [B, half_dim]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]


# The Flow Matching model encoding time t using a set of sine/cosine functions at different frequencies
class FlowMatching(nn.Module):
    def __init__(self, input_dim, time_embed_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)  # [B, time_embed_dim]
        x_t = torch.cat([x, t_emb], dim=1)
        return self.model(x_t)

def train_flow_matching(model, train_loader, epochs=50, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            t = torch.rand(batch_x.shape[0], 1, device=batch_x.device)
            f_xt = model(batch_x, t)

            dx_dt = -batch_x
            loss = criterion(f_xt, dx_dt)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10000 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

    return model

def determine_FMAD_hyperparameters(dataset_name_raw):
    dataset_name = dataset_name_raw.lower()
    if "census" in dataset_name:
        epoch_size = 5
        batch_size = 1024
        learning_rate = 0.005
    elif "backdoor" in dataset_name:
        epoch_size = 200
        batch_size = 1024
        learning_rate = 0.005
    elif "campaign" in dataset_name:
        epoch_size = 50
        batch_size = 1024
        learning_rate = 0.005
    elif "mnist" in dataset_name:
        epoch_size = 500
        batch_size = 512
        learning_rate = 0.005
    elif "speech" in dataset_name:
        epoch_size = 500
        batch_size = 512
        learning_rate = 0.005
    elif "optdigits" in dataset_name:
        epoch_size = 2000
        batch_size = 512
        learning_rate = 0.005
    elif "spambase" in dataset_name:
        epoch_size = 5000
        batch_size = 512
        learning_rate = 0.005
    elif "musk" in dataset_name:
        epoch_size = 5
        batch_size = 512
        learning_rate = 0.005
    elif "internetads" in dataset_name:
        epoch_size = 50
        batch_size = 512
        learning_rate = 0.005
    elif "donors" in dataset_name:
        epoch_size = 30
        batch_size = 1024
        learning_rate = 0.005
    elif "http" in dataset_name:
        epoch_size = 100
        batch_size = 1024
        learning_rate = 0.005
    elif "cover" in dataset_name:
        epoch_size = 10
        batch_size = 1024
        learning_rate = 0.005
    elif "fraud" in dataset_name:
        epoch_size = 75
        batch_size = 1024
        learning_rate = 0.005
    elif "skin" in dataset_name:
        epoch_size = 110  # Chosen first from "100 or 1"
        batch_size = 1024
        learning_rate = 0.005
    elif "celeba" in dataset_name:
        epoch_size = 2  # Chosen first from "100 or 1"
        batch_size = 1024
        learning_rate = 0.005
    elif "smtp" in dataset_name:
        epoch_size = 2
        batch_size = 1024
        learning_rate = 0.005
    elif "aloi" in dataset_name:
        epoch_size = 100
        batch_size = 1024
        learning_rate = 0.005
    elif "shuttle" in dataset_name:
        epoch_size = 200
        batch_size = 1024
        learning_rate = 0.005
    elif "magic.gamma" in dataset_name:
        epoch_size = 10
        batch_size = 1024
        learning_rate = 0.005
    elif "mammography" in dataset_name:
        epoch_size = 20
        batch_size = 1024
        learning_rate = 0.005
    elif "annthyroid" in dataset_name:
        epoch_size = 2000
        batch_size = 512
        learning_rate = 0.005
    elif "pendigits" in dataset_name:
        epoch_size = 1000
        batch_size = 512
        learning_rate = 0.005
    elif "satellite" in dataset_name:
        epoch_size = 10
        batch_size = 512
        learning_rate = 0.005
    elif "landsat" in dataset_name:
        epoch_size = 6
        batch_size = 512
        learning_rate = 0.005
    elif "satimage-2" in dataset_name:
        epoch_size = 5
        batch_size = 512
        learning_rate = 0.005
    elif "pageblocks" in dataset_name:
        epoch_size = 1800
        batch_size = 512
        learning_rate = 0.005
    elif "wilt" in dataset_name:
        epoch_size = 20
        batch_size = 512
        learning_rate = 0.005
    elif "thyroid" in dataset_name:
        epoch_size = 10
        batch_size = 512
        learning_rate = 0.005
    elif "waveform" in dataset_name:
        epoch_size = 580
        batch_size = 512
        learning_rate = 0.005
    elif "cardiotocography" in dataset_name:
        epoch_size = 1
        batch_size = 512
        learning_rate = 0.005
    elif "fault" in dataset_name:
        epoch_size = 5000
        batch_size = 512
        learning_rate = 0.005
    elif "cardio" in dataset_name:
        epoch_size = 2000
        batch_size = 512
        learning_rate = 0.005
    elif "letter" in dataset_name:
        epoch_size = 50
        batch_size = 512
        learning_rate = 0.005
    elif "yeast" in dataset_name:
        epoch_size = 130
        batch_size = 512
        learning_rate = 0.005
    elif "vowels" in dataset_name:
        epoch_size = 20
        batch_size = 512
        learning_rate = 0.005
    elif "pima" in dataset_name:
        epoch_size = 5
        batch_size = 512
        learning_rate = 0.005
    elif "breastw" in dataset_name:
        epoch_size = 1
        batch_size = 512
        learning_rate = 0.005
    elif "wdbc" in dataset_name:
        epoch_size = 2
        batch_size = 512
        learning_rate = 0.005
    elif "ionosphere" in dataset_name:
        epoch_size = 10
        batch_size = 512
        learning_rate = 0.005
    elif "stamps" in dataset_name:
        epoch_size = 200
        batch_size = 512
        learning_rate = 0.005
    elif "vertebral" in dataset_name:
        epoch_size = 25
        batch_size = 512
        learning_rate = 0.005
    elif "wbc" in dataset_name:
        epoch_size = 1
        batch_size = 512
        learning_rate = 0.005
    elif "glass" in dataset_name:
        epoch_size = 200
        batch_size = 512
        learning_rate = 0.005
    elif "wpbc" in dataset_name:
        epoch_size = 6
        batch_size = 512
        learning_rate = 0.005
    elif "lymphography" in dataset_name:
        epoch_size = 3
        batch_size = 512
        learning_rate = 0.005
    elif "wine" in dataset_name:
        epoch_size = 20
        batch_size = 512
        learning_rate = 0.005
    elif "hepatitis" in dataset_name:
        epoch_size = 1
        batch_size = 512
        learning_rate = 0.005

    return {
        "epochs": epoch_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size
            }