import torch
from VAE.model import VAENet
from VAE.helper import train as VAEtrain
from VAE.helper import compose as VAEcompose
from data import DataLoader, MyNote


def Loader_to_VAE_Test():
    input_depth = 42
    model = VAENet(input_depth,
                   section_length=1,
                   encoder_size=128,
                   decoder_size=64,
                   z_size=32,
                   conductor_size=64,
                   control_depth=32)
    loader = DataLoader()
    VAEtrain(model, loader, epoch_num=1, save="testVAE.model")


def VAE_compose_Test():
    model = torch.load("testVAE.model")
    section = torch.randn([1, 42])
    output = VAEcompose(model, section)
