import torch
import pickle
import numpy as np
import music21


def Generate_Melody_Notes(model, int2note_dict, network_input, length):
    random_index = np.random.randint(0, len(network_input) - 1)
    patten = network_input[random_index]
    result = [int2note_dict[patten[i][0]] for i in range(np.size(patten))]

    for i in range(length):  # 利用一段已有序列生成length个音符
        patten = np.reshape(patten, (np.size(patten), 1, 1))
        prediction_input = torch.torch.autograd.Variable(torch.Tensor(patten))
        prediction = model(prediction_input)
        prediction_noteindex = np.argmax(prediction.data.numpy())
        patten = patten[1:len(patten)]
        patten = np.insert(patten, len(patten), [prediction_noteindex])
        result.append(int2note_dict[prediction_noteindex])

    return result


def Create_midi(melody):
    Notes = []
    offset = 0
    for data in melody:
        note = music21.note.Note(data)
        note.offset = offset
        offset += 1
        note.storedInstrument = music21.instrument.Piano()
        Notes.append(note)
    midi_stream=music21.stream.Stream(Notes)
    midi_stream.write('midi', fp='demos/output1.mid')


if __name__ == "__main__":
    model=torch.load('Melody_LSTM\model_1857.pkl')
    model = model.to(device)
    model=model.eval()
    with open('int2note_dict.file','rb') as f:
        int2note_dict=pickle.load(f)
    with open('networkinput.file','rb') as f:
        network_input=pickle.load(f)
    
    melody=Generate_Melody_Notes(model,int2note_dict,network_input,16)
    Create_midi(melody)