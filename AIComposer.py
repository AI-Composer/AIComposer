import torch
import random
import json

from VAE.model import VAENet
from VAE.helper import train as VAEtrain
from VAE.helper import compose as VAEcompose

from data.test import all_test

with open("hparams.json", 'r') as f:
    hparams = json.load(f)

if __name__ == "__main__":
    all_test()
