import os
import pickle
import torch
import music21
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.LSTM=nn.LSTM(42, hidden_size, num_layers=Layer_Num, dropout=0.2)
        self.Dense1=nn.Linear(hidden_size, dense_hidden)
        self.Dense2=nn.Linear(dense_hidden, 42)
        self.softmax_p = nn.Softmax(dim=2)
        self.softmax_d = nn.Softmax(dim=2)
    
    def forward(self,x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x,_=self.LSTM(x) 
        #s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        #x = x.view(s*b, h)
        #print(x.size())
        # x=x[-1]  #取序列的最后一个，用来预测
        
        x=self.Dense1(x)
        x=self.Dense2(x)
        # print(x.size())
        pitch = x.index_select(2, torch.arange(0,29).to(device))
        duration = x.index_select(2, torch.arange(29,41).to(device))
        volume = x.index_select(2, torch.arange(41,42).to(device))
        # print(pitch.size())
        return (pitch, duration, volume)

    def train_model(self, dataLoader):
        batches, targets = dataLoader.getBatches()
        self.train()
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

        for epoch in range(epoches):
            for batch_idx in range(len(batches)):
                batch = batches[batch_idx]
                target = targets[batch_idx]
                batch.to(device)
                target.to(device)
                # print(batch.shape, ")))))))")
                L = batch.shape[0]
                x = torch.autograd.Variable(batch)
                x = x.to(device)
                self.optimizer.zero_grad()
                (pitch, duration, volume) = self.forward(x)
                pitch = pitch.view((L, 29))
                duration = duration.view((L, 12))
                volume = volume.view((L, 1))
                self.to(device)
                # target_p = (batch.index_select(2, torch.arange(0,29).to(device))).view((L, 29))
                # target_d = (batch.index_select(2, torch.arange(29,41).to(device))).view((L, 12))
                # target_v = (batch.index_select(2, torch.arange(41,42).to(device))).view((L, 1))
                target_p = target[:,0].view((L,)).to(device,dtype=torch.int64)
                target_d = target[:,1].view((L,)).to(device,dtype=torch.int64)
                target_v = target[:,2].view((L,)).to(device,dtype=torch.int64)

                Loss = F.cross_entropy(pitch, target_p) + F.cross_entropy(duration, target_d) # + F.mse_loss(volume, target_v)
                Loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                        epoch, batch_idx, len(batches),
                        100. * batch_idx / len(batches), Loss.item()))
            torch.save(self,'Melody_LSTM_v2/model.pkl')
            print("model saved!")

if __name__ == '__main__':
    model = MelodyLSTM()
    dataLoader = DataLoader(pickle_file='./Sequences.pkl')
    # dataLoader.split_transpose(1)
    model.train_model(dataLoader)