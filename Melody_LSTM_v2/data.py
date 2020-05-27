import os
import music21
import torch
import pickle


major_list = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]
minor_list = [0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6]
major_list_r = [0, 2, 4, 5, 7, 9, 11]
minor_list_r = [0, 2, 3, 5, 7, 8, 10]
durations = [0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

Sequences_pickle = 'Sequences.pkl'
test_midi = 'output1.mid'

class MyNote:
    def __init__(self,pitchID=0,duration=0,volume=0):
        self.pitchID=pitchID    # 实际上记录的是在调中的音级，即step
        self.duration=duration
        self.volume=volume
        #取0表示休止

class DataLoader:
    """
    初始化后在训练时调用getBatches来得到格式化的训练数据
    """
    def __init__(self, file_dir = 'simple_data', max_midi_num = None, pickle_file = 'Sequences_intervel_1.pkl'):
        """
        if max_midi_num = None, it will get all the midi files in the given dir, or will only get in max_midi_num midi files

        输入一个文件地址和间隔，输出一个list，list里的每个元素就是一个音符，这里为了李为方便使用，所以没有使用music21的note类，而是自己定义了一个mynote类。和弦的话就是直接也认为是一个note，只是它的pitch变成了多个音名。
        输出的list中每个元素有.pitch:格式为int，距离调式主音的音级，.duration：float,时长， .volume：float，力度
        """
        self.file_dir = file_dir
        self.max_midi_num = max_midi_num
        self.pickle_file = pickle_file
    
    def split_transpose(self, Split_Interval):
        i = 0
        files=os.listdir(self.file_dir)
        Sequences = []
        for file in files:
            i += 1
            if self.max_midi_num != None and i > self.max_midi_num:
                break
            stream=music21.converter.parse(os.path.join(self.file_dir, file))
            print(os.path.join(self.file_dir, file))

            # debug
            # stream = music21.converter.parse(test_midi)

            instru=music21.instrument.partitionByInstrument(stream)
            if instru:  # 如果有乐器部分，取第一个乐器部分            
                stream = instru.parts[0]
            else:
                stream = stream[0]
            key = stream.analyze('key')
            print(stream[0].name, key.name)

            timeSignature = stream[1]
            notes = stream[2:]
            tonicID = key.tonic.ps

            Tot_time=float(notes[-1].offset)+float(notes[-1].quarterLength)
            Sequence_len=int(Tot_time/Split_Interval+1)

            Sequence=[MyNote() for _ in range(Sequence_len)]#空序列
            for element in notes:
                index=int(element.offset/Split_Interval)
                if isinstance(element,music21.note.Note):
                    step = getStep(element.pitch.ps, tonicID, key.name.split(" ")[1])
                    Sequence[index]=MyNote(step, element.quarterLength, element.volume.getRealized())
                # elif (element, music21.chord.Chord):
                #     pitches='.'.join(str(n) for n in element.pitches)
                #     Sequence[index]=MyNote(pitches,element.quarterLength,element.volume.getRealized())

            # for ii in Sequence:
            #     print(ii.pitchID, ii.duration, ii.volume)

            Sequences.append(Sequence)
        with open(Sequences_pickle, 'wb') as f:
            pickle.dump(Sequences, f)
        print("successfully dumped in ", Sequences_pickle)

    def getBatches(self):
        with open(self.pickle_file, 'rb') as f:
            Sequences = pickle.load(f)
        batches = []
        targets = []
        for seq in Sequences:
            batch = []
            target = []
            for note in seq:
                pitch_list = [0 for i in range(29)]
                duration_list = [0 for i in range(12)]
                pitch_list[int(note.pitchID)] = 1
                duration_list[getDurationIndex(note.duration)] = 1
                features = pitch_list + duration_list + [note.volume]
                batch.append([features])
                target.append([int(note.pitchID) + 14, getDurationIndex(note.duration), note.volume])
            batch = torch.tensor(batch)
            target = torch.tensor(target)
            batches.append(batch)
            targets.append(target)

        
        return batches, targets



def getDurationIndex(duration):
    for index in range(12):
        if durations[index] == duration:
            return index
    return 0

def getStep(noteID, tonicID, tonality):
    """
    noteID and tonicID are both 'ps' attribute in music21;
    tonality is "major" or "minor"
    then return the step in the tonality
    """

    octave, delta = divmod((noteID - tonicID), 12)
    if tonality == 'major':
        if delta in {1, 3, 6, 8, 10}:
            print("not in tonality warning!")
        return octave * 7 + major_list[int(delta)]
    
    else:
        if delta in {1, 4, 6, 9, 11}:
            print("not in tonality warning!")
        return octave * 7 + minor_list[int(delta)]

def getPS(tonicID, tonality, step):
    """
    tonicID is 'ps' attribute in music21;
    tonality is "major" or "minor";
    step is the step in the tonality...
    """

    octave, delta = divmod(step, 7)
    if tonality == 'major':
        return tonicID + 12 * step + major_list_r[int(delta)]
    else:
        return tonicID + 12 * step + minor_list_r[int(delta)]

if __name__ == '__main__':
    dataLoader = DataLoader('simple_data')
    dataLoader.split_transpose(1)