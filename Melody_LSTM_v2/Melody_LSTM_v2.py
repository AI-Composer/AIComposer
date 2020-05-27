import os
import pickle
import torch
import music21
import torch.nn as nn
from data import dataLoader

Notes_Num = 29  # 上下两个八度 1 + 2 * 2 * 7
duration_Num = 12   # 一共12种duration

hidden_size = 512
dense_hidden = 512

epoches = 1

device = torch.device('cuda')

class MelodyLSTM(nn.Module):
    def __init__(self, Notes_Num, Layer_Num=2):
        super(MelodyLSTM, self).__init__()
        self.LSTM=nn.LSTM(3, hidden_size, num_layers=Layer_Num, dropout=0.2)
        self.Dense1=nn.Linear(hidden_size, dense_hidden)
        self.Dense2=nn.Linear(dense_hidden, 42)
        self.softmax=nn.Softmax()
    
    def forward(self,x):
        x,_=self.LSTM(x)
        #s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        #x = x.view(s*b, h)
        #print(x.size())
        # x=x[-1]  #取序列的最后一个，用来预测
        print(x.size())
        x=self.Dense1(x)
        x=self.Dense2(x)
        out=self.softmax(x)
        return out

    def train(self, dataLoader):
        batches = dataLoader.getBatches()
        for epoch in epoches:
            for batch_idx, batch in enumerate(batches):
                batch = batch.to(device)
