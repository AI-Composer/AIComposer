import os
import pretty_midi

input_dir = './MidiData/lmd_matched'    # Lakh dataset
output_dir = './MidiData/lead'

def getLeadTrack(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    

def main():
    for root, dirs, files in os.walk(input_dir):
        for midiFile in files:
            if midiFile == '.DS_Store':
                continue
            file_path = os.path.join(root, midiFile)
            getLeadTrack(file_path)

if __name__ == '__main__':
    main()