import music21


class mynote:
    def __init__(self,pitch=0,duration=0,volume=0):
        self.pitch=pitch
        self.duration=duration
        self.volume=volume
        #取0表示休止

def SplitMidi2Sequence(filepath,Split_Interval):
    #输入一个文件地址和间隔，输出一个list，list里的每个元素就是一个音符，这里为了李为方便使用，所以没有使用music21的note类，而是自己定义了一个mynote类。和弦的话就是直接也认为是一个note，只是它的pitch变成了多个音名。
    #输出的list中每个元素有.pitch:格式为str，音名，.duration：float,时长， .volume：float，力度

    stream=music21.converter.parse(filepath)
    instru=music21.instrument.partitionByInstrument(stream)
    if instru:  # 如果有乐器部分，取第一个乐器部分            
        notes = instru.parts[0].recurse()            
    else:  #如果没有乐器部分，直接取note                
        notes = stream.flat.notes
    Tot_time=float(notes[-1].offset)+float(notes[-1].quarterLength)
    Sequence_len=int(Tot_time/Split_Interval+1)
    rest=mynote() #休止符
    
    Sequence=[None for _ in range(Sequence_len)]#空序列
    for element in notes:
        index=int(element.offset/Split_Interval)
        if isinstance(element,music21.note.Note):
            Sequence[index]=mynote(str(element.pitch),element.quarterLength,element.volume.getRealized())
        elif (element, music21.chord.Chord):
            pitches='.'.join(str(n) for n in element.pitches)
            Sequence[index]=mynote(pitches,element.quarterLength,element.volume.getRealized())

    return Sequence
if __name__ == "__main__":
    file="output1.mid"
    a=SplitMidi2Sequence(file,1/6)
    print([i.pitch for i in a])
    input()