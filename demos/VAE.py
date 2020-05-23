import torch
import torch.nn as nn


class VAENet(nn.Module):
    """A really simplified demo of VAE net"""
    def __init__(self, sequence_length):
        self.bidlstm = nn.LSTM(sequence_length)