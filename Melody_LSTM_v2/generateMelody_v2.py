import os
import pickle
import torch
import torch.nn as nn
import music21
from Melody_LSTM_v2 import MelodyLSTM
from data import *
import time

tune = 'E minor'
default_model = "Melody_LSTM_v2/model_0.pkl"
Sequence_length = 64

device = torch.device('cuda')

def generateMelody(model_file=default_model):
    """
    生成一个我们定义的Sequence序列
    """
    model = torch.load(model_file)
    model = model.to(device)
    
    Sequence = [MyNote(0, 1, 0.7)]
    for note_idx in range(1, Sequence_length):  # 指生成第几个音
        batch = []
        for note in Sequence[:note_idx]:
            pitch_list = [0 for i in range(29)]
            duration_list = [0 for i in range(12)]
            pitch_list[int(note.pitchID)] = 1
            duration_list[getDurationIndex(note.duration)] = 1
            features = pitch_list + duration_list + [note.volume]
            batch.append([features])
        batch = torch.Tensor(batch)
        batch = batch.to(device)

        #print("_________________", model(batch))

        (pitch, duration, volume) = model(batch)
        pitch = pitch.view((note_idx, 29))
        duration = duration.view((note_idx, 12))
        pitches = torch.argmax(pitch, dim=1)
        ds = torch.argmax(duration, dim=1)
        new_note = MyNote(pitches[-1].item()-14, durations[ds[-1].item()], 0.7)
        Sequence.append(new_note)

    return Sequence

def createMidi(Sequence, split_interval=1, output_file='./demos/cminor.mid', tune=tune):
    """
    根据Sequence生成对应调式的midi
    """
    tonic = tune.split(" ")[0]
    mode = tune.split(" ")[1]
    key = music21.key.Key(tonic)
    tonicID = key.tonic.ps
    stream = music21.stream.Stream()

    offset = 0
    for mynote in Sequence:
        p = music21.pitch.Pitch()
        p.ps = getPS(tonicID, mode, mynote.pitchID)
        new_note = music21.note.Note(p)
        new_note.offset = offset
        #new_note.pitch.ps = getPS(tonicID, mode, mynote.pitchID)
        new_note.duration.quarterLength = mynote.duration
        stream.insert(offset, new_note)
        offset += split_interval
    stream.write('midi', fp=output_file)
    print("successfully written in ", output_file)
    stream.show('text')

if __name__ == '__main__':
    new_seq = generateMelody()
    createMidi(new_seq, output_file = './demos/'+tune+str(time.time())+'.mid')