import GetModel
import GetData
import numpy as np
import torch
import pickle

device = torch.device('cuda')

def train(filepath,batch_size,turn):
    Notes=GetData.GetNotes_Melody(filepath)
    Notes_reshape=[]   #因为set不能对多层嵌套列表作用，所以先把它压平
    for note in Notes:
        Notes_reshape+=note
    Notes_Num=len(set(Notes_reshape))
    Note_name=sorted(set(Notes_reshape)) #获得排序的不重复的音符名字
    sequence_length=15        #序列长度
    note2int_dict=dict((j,i) for i,j in enumerate(Note_name))  #设计一个字典，把音符转换成数字，方便训练
    int2note_dict=dict((i,j) for i,j in enumerate(Note_name))  #一个字典，把数字转换回音符
    with open('int2note_dict.file','wb') as f:
        pickle.dump(int2note_dict,f)
    network_input=[]#输入序列
    network_output=[]#输出序列
    for Note in Notes:
        for i in range(0, len(Note) - sequence_length):
            #每输入sequence_length个音符，输出一个音符
            sequence_in=Note[i: i + sequence_length]
            sequence_out=Note[i + sequence_length]
            network_input.append([note2int_dict[k] for k in sequence_in])
            network_output.append(note2int_dict[sequence_out])
    network_input=np.reshape(network_input, (len(network_input), sequence_length, 1))
    with open('networkinput.file','wb') as f:
        pickle.dump(network_input,f)
    model=GetModel.MelodyLSTM(Notes_Num).to(device)   #网络inputsize=1，输出维数为1
    Loss_Func=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.RMSprop(model.parameters(),lr=1e-3,alpha=0.9)
    for j in range(turn):
        for i in range(int(len(network_input)/batch_size)):
            model.zero_grad() #清空梯度缓存
            data=np.array(network_input[i:batch_size+i])
            data=np.reshape(data.transpose(),(sequence_length,batch_size,1))    #(seq_len,batch_size,input_size)
            #print(data.shape)
            input1=torch.autograd.Variable(torch.Tensor(data))
            #targets=torch.autograd.Variable(torch.Tensor([int(network_output[i])]))
            #targets =torch.LongTensor(targets)
            targets_data=np.array(network_output[i:i+batch_size])
            targets=torch.tensor(targets_data.transpose(), dtype=torch.long)
            targets = targets.to(device)
            input1 = input1.to(device)
            output=model(input1)
            loss=Loss_Func(output,targets)
            loss.backward()
            optimizer.step()
            if (j*int(len(network_input)/batch_size)+i+1) % 10 == 0: 
                print('Epoch: {}, Loss:{:.5f}'.format(j*int(len(network_input)/batch_size)+i+1, loss.item())) 
    torch.save(model,'Melody_LSTM/model_1857.pkl')
    return 0

if __name__ == "__main__":
    filepath='./simple_data/'
    train(filepath,20,10)  #(filepath,batch_size,turn)