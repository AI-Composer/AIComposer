import music21
import os
import torch

def GetNotes_Melody(filepath):
    files=os.listdir(filepath)
    Notes=[]
    i=0
    for file in files:
        Note=[]
        print(filepath+file)
        try:
            i=i+1
            if(i==50):
                break
            stream=music21.converter.parse(filepath+file)
            instru=music21.instrument.partitionByInstrument(stream)
            if instru:  # 如果有乐器部分，取第一个乐器部分            
                notes = instru.parts[0].recurse()            
            else:  #如果没有乐器部分，直接取note                
                notes = stream.flat.notes
            for element in notes:                
                # 如果是 Note 类型，取音调                
                #如果是 Chord 类型，取音调的序号,存int类型比较容易处理                
                if isinstance(element, music21.note.Note): 
                    #print(str(element.pitch))                   
                    Note.append(str(element.pitch))                
                #elif isinstance(element, music21.chord.Chord):    #这里作为旋律的训练先不要和弦
                    #print(element.normalOrder)                    
                    #Note.append('.'.join(str(n) for n in element.normalOrder))
            Notes.append(Note)
        except:
            pass

        #print(Notes)
        #with open(filepath+file+'_note','a+') as f:
        #    f.write(str(Notes))
    return Notes
if __name__ == "__main__":
    
    filepath='./simple'
    GetNotes_Melody(filepath)