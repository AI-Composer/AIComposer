import torch
import torch.nn as nn
import GetData

class MelodyLSTM(nn.Module):
    def __init__(self,Notes_Num,Layer_Num=3):
        super(MelodyLSTM, self).__init__()
        self.LSTM=nn.LSTM(1,512,num_layers=Layer_Num,dropout=0.2)
        self.Dense1=nn.Linear(512,512)
        self.Dense2=nn.Linear(512,Notes_Num)
        self.softmax=nn.Softmax()
    
    def forward(self,x):
        #print(x.size())
        x,_=self.LSTM(x)
        #s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        #x = x.view(s*b, h)
        #print(x.size())
        x=x[-1]  #取序列的最后一个，用来预测
        #print(x.size())
        x=self.Dense1(x)
        x=self.Dense2(x)
        out=self.softmax(x)
        return out

if __name__ == "__main__":
    Notes=GetData.GetNotes_Melody('./simple')
    a=MelodyLSTM(100)
    print(a)


