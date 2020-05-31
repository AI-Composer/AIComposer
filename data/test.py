import torch

from data.midi import midis_to_sequences, sequences_to_midis
from data.format import getBatches, sequences_to_tensors
from data.save import save, load

from util.logging import logger


def midis_to_sequences_test():
    logger.info("midis_to_sequences_test() started")
    sequences = midis_to_sequences('/home/hades/Documents/simple_data')
    assert isinstance(sequences, list)
    assert isinstance(sequences[0], torch.Tensor)
    assert sequences[0].size()[0] == 3
    assert sequences[0].size()[2] == 3
    logger.info("midis_to_sequences_test() passed")


def sequences_to_midis_test():
    logger.info("sequences_to_midis_test() started")
    sequences = midis_to_sequences('/home/hades/Documents/simple_data')
    sequences_to_midis(sequences)
    logger.info("sequences_to_midis_test() passed")


def getBatches_test():
    logger.info("getBatches_test() started")
    sequences = midis_to_sequences('/home/hades/Documents/simple_data')
    batches, targets, batch_size = getBatches(sequences)
    assert isinstance(batches, list)
    assert isinstance(batches[0], torch.Tensor)
    assert batches[0].size()[0] == 3
    assert batches[0].size()[2] == batch_size
    assert batches[0].size()[3] == 42
    assert isinstance(targets, list)
    assert isinstance(targets[0], torch.Tensor)
    assert targets[0].size()[0] == 3
    assert targets[0].size()[2] == batch_size
    assert targets[0].size()[3] == 3
    tensors = sequences_to_tensors(sequences)
    batches, targets, batch_size = getBatches(tensors)
    assert isinstance(batches, list)
    assert isinstance(batches[0], torch.Tensor)
    assert batches[0].size()[0] == 3
    assert batches[0].size()[2] == batch_size
    assert batches[0].size()[3] == 42
    assert isinstance(targets, list)
    assert isinstance(targets[0], torch.Tensor)
    assert targets[0].size()[0] == 3
    assert targets[0].size()[2] == batch_size
    assert targets[0].size()[3] == 3
    logger.info("getBatches_test() passed")


def save_test():
    logger.info("save_test() started")
    sequences = midis_to_sequences('/home/hades/Documents/simple_data')
    path = save(sequences, 'sequences.test')
    sequences = load(path)
    assert isinstance(sequences, list)
    assert isinstance(sequences[0], torch.Tensor)
    assert sequences[0].size()[0] == 3
    assert sequences[0].size()[2] == 3
    logger.info("save_test() passed")


def all_test():
    sequences_to_midis_test()
    getBatches_test()
    save_test()
