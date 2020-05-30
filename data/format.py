import torch


def sequence_to_tensor(sequence):
    """Helper function to convert sequence to tensor
    Args:
        sequence: torch.Tensor, [sequence_length, 3, 3]
    """


def Seq_to_noteseq(output):
    pitch_output = output[:, :29, 0]
    duration_output = output[:, 29:41, 0]
    volume_output = output[:, 41, 0]
    pitch_output = torch.argmax(pitch_output, dim=1)
    duration_output = torch.argmax(duration_output, dim=1)

    sequence = []
    for i in range(output.size()[0]):
        sequence.append(
            MyNote(pitchID=int(pitch_output[i] - 14),
                   duration=durations[duration_output[i]],
                   volume=volume_output[i]))
    return sequence


def getBatches_1(self):
    with open(self.pickle_file, 'rb') as f:
        Sequences = pickle.load(f)
    batches = []
    targets = []
    for seq in Sequences:
        batch = []
        target = []
        for note in seq:
            features = torch.zeros([42], dtype=torch.float32)
            features[int(note.pitchID)] = 1
            features[29 + getDurationIndex(note.duration)] = 1
            features[-1] = note.volume
            batch.append(features.unsqueeze(dim=0).unsqueeze(dim=-1))
            target_features = torch.tensor([
                note.pitchID + 14,
                getDurationIndex(note.duration), note.volume
            ],
                                           dtype=torch.float32)
            target.append(target_features.unsqueeze(dim=0).unsqueeze(dim=-1))
        batch = torch.stack(batch, dim=0)
        target = torch.stack(target, dim=0)
        batches.append(batch)
        targets.append(target)
    return batches, targets
