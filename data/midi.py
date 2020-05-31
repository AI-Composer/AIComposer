import os
import time

import torch
import music21
from tqdm import tqdm

from util.logging import logger

major_list = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]
minor_list = [0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6]
major_list_r = [0, 2, 4, 5, 7, 9, 11]
minor_list_r = [0, 2, 3, 5, 7, 8, 10]
duration_list = [
    0.0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0
]


def __get_duration_index__(quarter_length):
    """Helper function for getting the scalar duration index
       ****************************************************
       For safety, DON'T call this function from outside!!!
       ****************************************************
    Args:
        quarter_length: music21.note.Note.quarterLength, float-like
    Returns:
        index: int
    """
    for index, duration in enumerate(duration_list):
        if quarter_length - duration < 0.01:
            return index
    logger.error("duration not in list! got {}".format(quarter_length))
    exit()


def sequence_to_midi(sequence,
                     split_interval=1,
                     tune='E minor',
                     folder=None,
                     name=None):
    """Create one midi from Tensor
       NOW PATH IS FIXED TO outputs/
    Args:
        sequence: torch.Tensor, [3, sequence_length, 3]
        split_interval: scalar, indicates the frequency of notes on reconstruction
        tune: str, 'X xxxxx', 'E minor', etc
        folder: str, folder to save midis
        name: str, name of midi file
    Returns:
        None
    """
    assert isinstance(sequence,
                      torch.Tensor), "wrong sequence class, got {}".format(
                          sequence.__class__.__name__)
    assert sequence.dim() == 3 and sequence.size()[0] == 3 and sequence.size(
    )[2] == 3, "wrong sequence shape, got {}".format(sequence.size())
    # Part1: Get correct parameters
    standard_ps = music21.key.Key(tune.split(" ")[0]).tonic.ps
    mode = tune.split(" ")[1]
    if mode == 'minor':
        ps_list = minor_list_r
    elif mode == 'major':
        ps_list = major_list_r
    else:
        logger.warning(
            "unsupported tune mode! got {}, will continue with 'major'".format(
                mode))
        ps_list = major_list_r

    stream = music21.stream.Stream()
    if folder is None:
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        logger.warning(
            "folder name not assigned, will use current time {}".format(rq))
        folder = 'outputs/' + rq
    if not os.path.exists(folder):
        os.mkdir(folder)
    if name is None:
        name = str(hash(time.time()))
    if name[-4:] != '.mid':
        name = name + '.mid'
    path = os.path.join(folder, name)

    # Part2: Analyse sequence information
    # FIXME So far only lead track is created
    lead_track = sequence[0, :, :]
    offset = 0
    for note in lead_track:
        # Decide ps
        p = music21.pitch.Pitch()
        octave, delta = divmod(int(note[0]), 7)
        p.ps = standard_ps + 12 * octave + ps_list[int(delta)]
        # Decide offset and duration
        new_note = music21.note.Note(p)
        new_note.offset = offset
        new_note.duration.quarterLength = duration_list[int(note[1])]
        # FIXME So far no volume information converted?
        stream.insert(offset, new_note)
        offset += split_interval

    # Part3: write output file
    stream.write('midi', fp=path)


def sequences_to_midis(sequences,
                       split_interval=1,
                       tune='E minor',
                       folder=None):
    """Create midis from Tensor, most compatitive verson
       NOW PATH IS FIXED TO outputs/
    Args:
        tensor: list of torch.Tensor, [total_num, 3, sequence_length, 3]
        split_interval: scalar, indicates the frequency of notes on reconstruction
        tune: str, 'X xxxxx', 'E minor', etc
        folder: str, folder to save midis
    Returns:
        None
    """
    assert isinstance(sequences, list), "wrong sequences class, got {}".format(
        sequences.__class__.__name__)
    logger.info("{} midis to create".format(len(sequences)))
    # create folder if None
    if folder is None:
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        logger.warning(
            "folder name not assigned, will use current time {}".format(rq))
        folder = 'outputs/' + rq
    if not os.path.exists(folder):
        os.mkdir(folder)
    # Enter main loop
    for num, sequence in tqdm(enumerate(sequences), total=len(sequences)):
        sequence_to_midi(sequence,
                         split_interval,
                         tune,
                         folder,
                         name=str(num) + '.mid')
    logger.info("Midi creation completed")


def midi_to_sequence(filepath, split_interval=1):
    """Create sequence from midi file, using music21 and torch
    Args:
        filepath: str, full or relative file path
        split_interval: int
    Returns:
        sequence: torch.Tensor, [3, sequence_length, 3]
    """
    stream = music21.converter.parse(filepath)
    instru = music21.instrument.partitionByInstrument(stream)
    # If instrument part exists, select it, else select the first stream
    if instru:
        stream = instru.parts[0]
    else:
        stream = stream[0]

    # set up parameters
    key = stream.analyze('key')
    tonicID = key.tonic.ps
    mode = key.name.split(" ")[1]
    if mode == 'minor':
        step_list = minor_list
        tonality_ban = [1, 4, 6, 9, 11]
    elif mode == 'major':
        step_list = major_list
        tonality_ban = [1, 3, 6, 8, 10]
    else:
        logger.warning(
            "unsupported tune mode! got {}, will continue with 'major'".format(
                mode))
        step_list = major_list
        tonality_ban = [1, 3, 6, 8, 10]
    # FIXME why start from 2?
    notes = stream[2:]
    tot_time = float(notes[-1].offset) + float(notes[-1].quarterLength)
    sequence_len = int(tot_time / split_interval + 1)
    sequence = torch.zeros([3, sequence_len, 3], dtype=torch.float32)

    # Enter main loop
    for note in notes:
        index = int(note.offset / split_interval)
        if isinstance(note, music21.note.Note):
            octave, delta = divmod((note.pitch.ps - tonicID), 12)
            delta = int(delta)
            if delta in tonality_ban:
                logger.warning(
                    "note not in your chosen tonality! got {}".format(delta))
            step = octave * 7 + step_list[delta]
            # FIXME now the chord and drum tracks are set to zeros
            sequence[0][index][0] = step
            sequence[0][index][1] = __get_duration_index__(note.quarterLength)
            sequence[0][index][2] = note.volume.getRealized()
    return sequence


def midis_to_sequences(folder, total_num=None, split_interval=1):
    """Create tensor from midi folder, using music21 and torch
    Args:
        folder: str, full or relative folder path
        total_num: int, if assigned, the first dimension of the output rensor will be
                   `min(num_of_midis, total_num)`
        split_interval: int
    Returns:
        tensor: list of torch.Tensor, [total_num, 3, sequence_length, 3]
                Different sequences will have different sequence_length
    """
    logger.info(
        "Prepare to create tensor from midis, folder {}".format(folder))

    # set up parameters
    filenames = os.listdir(folder)
    sequences = []
    # Enter main loop
    for count, filename in tqdm(enumerate(filenames), total=len(filenames)):
        if total_num is not None and count >= total_num:
            break
        filepath = os.path.join(folder, filename)
        sequences.append(midi_to_sequence(filepath, split_interval))
    logger.info("Creation completed.")
    return sequences
