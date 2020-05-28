import os
import pickle
import torch
import torch.nn as nn
import music21
from Melody_LSTM_v2 import MelodyLSTM

def generateMelody(model_file):
    model = torch.load("Melody_LSTM_v2/model.pkl")
