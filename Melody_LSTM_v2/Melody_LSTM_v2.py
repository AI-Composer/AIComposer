import os
import pickle
import torch
import music21
import torch.nn as nn
import torch.nn.functional as F
from data import DataLoader

Notes_Num = 29  # 上下两个八度 1 + 2 * 2 * 7
duration_Num = 12   # 一共12种duration

hidden_size = 512
dense_hidden = 512

epoches = 1

device = torch.device('cuda')

class MelodyLSTM(nn.Module):
    def __init__(self, Layer_Num=2):
        super(MelodyLSTM, self).__init__()
        self.LSTM=nn.LSTM(3, hidden_size, num_layers=Layer_Num, dropout=0.2)
        self.Dense1=nn.Linear(hidden_size, dense_hidden)
        self.Dense2=nn.Linear(dense_hidden, 42)
        self.softmax_p = nn.Softmax()
        self.softmax_d = nn.Softmax()
    
    def forward(self,x):
        x,_=self.LSTM(x)
        #s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        #x = x.view(s*b, h)
        #print(x.size())
        # x=x[-1]  #取序列的最后一个，用来预测
        print(x.size())
        x=self.Dense1(x)
        x=self.Dense2(x)
        pitch = self.softmax_p(x[:][:][:29])
        duration = self.softmax_d(x[:][:][29:41])
        volume = x[:][:][-1]
        return (pitch, duration, volume)

    def train(self, dataLoader):
        batches = dataLoader.getBatches()
        self.train()
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

        for epoch in epoches:
            for batch_idx, batch in enumerate(batches):
                L = batch.shape()[0]
                x = torch.autograd.Variable(batch)
                x = x.to(device)
                optimizer.zero_grad()
                (pitch, duration, volume) = self.forward(x)
                pitch = pitch.view((L, 29))
                duration = duration.view((L, 12))
                volume = volume.view((L, 1))

                target_p = (batch[:][:][:29]).view((L, 29))
                target_d = (batch[:][:][29:41]).view((L, 12))
                target_v = (batch[:][:][-1]).view((L, 1))

                Loss = F.cross_entropy(pitch, target_p) + F.cross_entropy(duration, target_d) + F.mse_loss(volume, target_d)

                Loss.backward()
                self.optimizer.step()
                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAccuracy: {:.2f}%'.format(
                        epoch, batch_idx, len(batch),
                        100. * batch_idx / len(batch), Loss.item()))
            torch.save(self,'Melody_LSTM_v2/model.pkl')

if __name__ == '__main__':
    model = MelodyLSTM()
    dataLoader = DataLoader('Sequences_intervel_1.pkl')
    model.train(dataLoader)