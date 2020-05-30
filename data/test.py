import torch

from data.midi import midis_to_sequences, sequences_to_midis
from data.format import getBatches, sequences_to_tensors
from data.save import save, load


def midis_to_sequences_test():
    sequences = midis_to_sequences('/home/hades/Documents/simple_data')
    assert isinstance(sequences, list)
    assert isinstance(sequences[0], torch.Tensor)
    assert sequences[0].size()[0] == 3
    assert sequences[0].size()[2] == 3


def sequences_to_midis_test():
    sequences = midis_to_sequences('/home/hades/Documents/simple_data')
    sequences_to_midis(sequences)


def getBatches_test():
    sequences = midis_to_sequences('/home/hades/Documents/simple_data')
    batches, batch_size = getBatches(sequences)
    assert isinstance(batches, list)
    assert isinstance(batches[0], torch.Tensor)
    assert batches[0].size()[0] == 3
    assert batches[0].size()[2] == batch_size
    assert batches[0].size()[3] == 42
    tensors = sequences_to_tensors(sequences)
    batches, batch_size = getBatches(tensors)
    assert isinstance(batches, list)
    assert isinstance(batches[0], torch.Tensor)
    assert batches[0].size()[0] == 3
    assert batches[0].size()[2] == batch_size
    assert batches[0].size()[3] == 42


def save_test():
    sequences = midis_to_sequences('/home/hades/Documents/simple_data')
    path = save(sequences, 'sequences.test')
    sequences = load(path)
    assert isinstance(sequences, list)
    assert isinstance(sequences[0], torch.Tensor)
    assert sequences[0].size()[0] == 3
    assert sequences[0].size()[2] == 3


def all_test():
    midis_to_sequences_test()
    sequences_to_midis_test()
    getBatches_test()
    save_test()
