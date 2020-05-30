import torch
import random
import json

from VAE.model import VAENet
from VAE.helper import train as VAEtrain
from VAE.helper import compose as VAEcompose
from Melody_LSTM_v2.data import DataLoader, MyNote, durations
from Melody_LSTM_v2.generateMelody_v2 import createMidi

from data.midi import getDurationIndex

with open("hparams.json", 'r') as f:
    hparams = json.load(f)


def generateMelody(model_file=hparams['LSTM']['best_model']):
    """
    生成一个我们定义的Sequence序列
    """
    model = torch.load(model_file)
    model = model.to(hparams['device'])

    Sequence = [MyNote(0, 1, 0.7)]
    for note_idx in range(1, hparams['LSTM']['output_length']):  # 指生成第几个音
        batch = []
        for note in Sequence[:note_idx]:
            pitch_list = [0 for i in range(29)]
            duration_list = [0 for i in range(12)]
            pitch_list[int(note.pitchID)] = 1
            duration_list[getDurationIndex(note.duration)] = 1
            features = pitch_list + duration_list + [note.volume]
            batch.append([features])
        batch = torch.Tensor(batch)
        batch = batch.to(hparams['device'])

        # print("_________________", model(batch))

        (pitch, duration, volume) = model(batch)
        pitch = pitch.view((note_idx, 29))
        duration = duration.view((note_idx, 12))
        pitches = torch.argmax(pitch, dim=1)
        ds = torch.argmax(duration, dim=1)
        new_note = MyNote(pitches[-1].item() - 14, durations[ds[-1].item()],
                          0.7)
        Sequence.append(new_note)

    return Sequence


def Loader_to_VAE_Test():
    input_depth = 42
    model = VAENet(input_depth,
                   section_length=1,
                   encoder_size=64,
                   decoder_size=32,
                   z_size=16,
                   conductor_size=32,
                   control_depth=16)
    loader = DataLoader()
    VAEtrain(model,
             loader,
             epoch_num=50,
             batch_num=5,
             save="testVAE.model",
             summary=True)


def VAE_compose_Test():
    model = torch.load("testVAE.model")
    section = torch.zeros([1, 42])
    section[0][14] = 1
    section[0][35] = 1
    section[0][41] = 0.7
    output = VAEcompose(model, section)

    createMidi(Seq_to_noteseq(output), output_file="testVAE.mid")


def Get_Batches_Test():
    loader = DataLoader()
    inputs, targets = loader.getBatches_1()
    sample = random.choice(inputs)
    sample = sample.squeeze().unsqueeze(-1)
    createMidi(Seq_to_noteseq(sample), output_file="testBatches.mid")


if __name__ == "__main__":
    Loader_to_VAE_Test()
    VAE_compose_Test()
