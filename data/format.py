import torch
from tqdm import tqdm

from util.logging import logger


def sequence_to_tensor(sequence):
    """Helper function to convert sequence to tensor
    Args:
        sequence: torch.Tensor, [3, sequence_length, 3]
    Returns:
        tensor: torch.Tensor, [3, sequence_length, 42]
    """
    assert isinstance(
        sequence, torch.Tensor), "input must be torch.Tensor, got {}".format(
            sequence.__class__.__name__)
    assert sequence.dim() == 3 and sequence.size()[0] == 3 and sequence.size(
    )[2] == 3, "invalid input shape, got {}".format(sequence.size())
    tensor = torch.zeros([3, sequence.size()[1], 42])
    # Enter main loop
    for track_idx, track in enumerate(sequence):
        for note_idx, note in enumerate(track):
            tensor[track_idx][note_idx][int(note[0])] = 1
            tensor[track_idx][note_idx][int(note[1]) + 29] = 1
            tensor[track_idx][note_idx][41] = note[2]
    return tensor


def tensor_to_sequence(tensor):
    """Helper function to convert tensor to sequence
    Args:
        tensor: torch.Tensor, [3, sequence_length, 42]
    Returns:
        sequence: torch.Tensor, [3, sequence_length, 3]
    """
    assert isinstance(
        tensor, torch.Tensor), "input must be torch.Tensor, got {}".format(
            tensor.__class__.__name__)
    assert tensor.dim() == 3 and tensor.size()[0] == 3 and tensor.size(
    )[2] == 42, "invalid input shape, got {}".format(tensor.size())
    pitch = torch.argmax(tensor[0, :, :29], dim=-1)
    duration = torch.argmax(tensor[0, :, 29:41], dim=-1)
    volume = tensor[0, :, 41]
    sequence = torch.stack((pitch, duration, volume), dim=-1)
    return sequence


def __LOOP__(inputs, func):
    """Helper function for first-dim-loop
       ****************************************************
       For safety, DON'T call this function from outside!!!
       ****************************************************
    Args:
        inputs: list, [total_num, ?]
    Returns:
        outputs: list, [total_num, ?]
    """
    outputs = []
    for item in tqdm(inputs):
        outputs.append(func(item))
    return outputs


def sequences_to_tensors(sequences):
    """Helper function to convert sequences to tensors
    Args:
        sequences: list of torch.Tensor, [total_num, 3, sequence_length, 3]
    Returns:
        tensors: list of torch.Tensor, [total_num, 3, sequence_length, 42]
    """
    assert isinstance(sequences, list), "input must be list, got {}".format(
        sequences.__class__.__name__)
    logger.info("Transfering sequences to tensors, length {}".format(
        len(sequences)))
    tensors = __LOOP__(sequences, sequence_to_tensor)
    logger.info("Transfer completed")
    return tensors


def tensors_to_sequences(tensors):
    """Helper function to convert tensors to sequences
    Args:
        tensors: list of torch.Tensor, [total_num, 3, sequence_length, 42]
    Returns:
        sequences: list of torch.Tensor, [total_num, 3, sequence_length, 3]
    """
    assert isinstance(tensors, list), "input must be list, got {}".format(
        tensors.__class__.__name__)
    logger.info("Transfering tensors to sequences, length {}".format(
        len(tensors)))
    sequences = __LOOP__(tensors, tensor_to_sequence)
    logger.info("Transfer completed")
    return sequences


def __cmp_seq_len__(tensor):
    """cmp use
       ****************************************************
       For safety, DON'T call this function from outside!!!
       ****************************************************
    Args:
        tensor: torch.Tensor, [3, sequence_length, 42]
    Returns:
        priority: int
    """
    return int(tensor.size()[1])


def __tensors_to_batches_batch_num__(tensors, batch_num):
    """'BATCH_NUM' mode
       ****************************************************
       For safety, DON'T call this function from outside!!!
       ****************************************************
    Args:
        tensors: list of torch.Tensor, [total_num, 3, sequence_length, 42]
        batch_num: int
    Returns:
        batches: list of torch.Tensor, [batch_num, 3, sequence_length, batch_size, 42]
        batch_size: int, result batch size
    """
    total_num = len(tensors)
    # chunk downside
    batch_size = int(total_num / batch_num)
    # sort small --- large
    tensors.sort(key=__cmp_seq_len__)
    batches = []
    # Enter main loop
    for idx in range(0, total_num, batch_size):
        batch = tensors[idx:idx + batch_size]
        # Must pad before convert to torch.Tensor
        max_sequence_length = max(batch, key=__cmp_seq_len__).size()[1]
        padded_batch = []
        for sequence in batch:
            sequence_length = sequence.size()[1]
            if sequence_length < max_sequence_length:
                sequence = torch.cat(
                    (sequence,
                     torch.zeros(
                         [3, max_sequence_length - sequence_length, 42])),
                    dim=1)
            padded_batch.append(sequence)
        padded_batch = torch.stack(padded_batch, dim=2)
        batches.append(padded_batch)
    return (batches, batch_size)


def __tensors_to_batches_keep_len__(tensors, tolorance):
    """'BATCH_NUM' mode
       ****************************************************
       For safety, DON'T call this function from outside!!!
       ****************************************************
    Args:
        tensors: list of torch.Tensor, [total_num, 3, sequence_length, 42]
        tolorance: int
    Returns:
        batches: list of torch.Tensor, [batch_num, 3, sequence_length, batch_size, 42]
        batch_size: int, result batch size
    """
    logger.error("__tensors_to_batches_keep_len__ not implemented yet")
    exit()


def tensors_to_batches(tensors, mode='BATCH_NUM', **kwargs):
    """Helper function to convert tensors to batches
    This function requires an important `mode` parameter, which controls the batch process
    This fuction gurantee sequence_length are the same within a batch
    Args:
        tensors: list of torch.Tensor, [total_num, 3, sequence_length, 42]
        mode: str, mode control
              'BATCH_NUM': DEFAULT, function will try to satisfy batch number
                           if necessary, some short sequences will be padded
                           when this mode selected, kwarg `batch_num` is availible
                           by default, batch_num will be 20
              'KEEP_LEN': function will try to keep sequence length,
                          if some length have only one sample, batch size will be 1
                          when this mode selected, kwarg `tolorance` is availible
                          by default, tolorance will be 0
    Returns:
        batches: list of torch.Tensor, [batch_num, 3, sequence_length, batch_size, 42]
        batch_size: int, result batch size
    """
    assert isinstance(tensors, list), "input must be list, got {}".format(
        tensors.__class__.__name__)
    logger.info("Transfering tensors to batches, length {}, mode {}".format(
        len(tensors), mode))
    if mode == 'BATCH_NUM':
        if 'batch_num' in kwargs.keys():
            batch_num = int(kwargs['batch_num'])
        else:
            logger.warning("batch_num not provided, using default 20")
            batch_num = 20
        batches, batch_size = __tensors_to_batches_batch_num__(
            tensors, batch_num)
    elif mode == 'KEEP_LEN':
        if 'tolorance' in kwargs.keys():
            tolorance = int(kwargs['tolorance'])
        else:
            logger.warning("tolorance not provided, using default 0")
            tolorance = 0
        batches, batch_size = __tensors_to_batches_keep_len__(
            tensors, tolorance)
    else:
        logger.warning(
            "invalid mode, using default 'BATCH_NUM' with batch_num=20")
        batches, batch_size = __tensors_to_batches_batch_num__(tensors, 20)
    logger.info("Transfer completed, batch_num {}".format(len(batches)))
    return (batches, batch_size)


def sequences_to_batches(sequences, mode='BATCH_NUM', **kwargs):
    """Helper function to convert sequences to batches
    This function requires an important `mode` parameter, which controls the batch process
    This fuction gurantee sequence_length are the same within a batch
    Args:
        sequences: list of torch.Tensor, [total_num, 3, sequence_length, 3]
        mode: str, mode control
              'BATCH_NUM': DEFAULT, function will try to satisfy batch number
                           if necessary, some short sequences will be padded
                           when this mode selected, kwarg `batch_num` is availible
                           by default, batch_num will be 20
              'KEEP_LEN': function will try to keep sequence length,
                          if some length have only one sample, batch size will be 1
                          when this mode selected, kwarg `tolorance` is availible
                          by default, tolorance will be 0
    Returns:
        batches: list of torch.Tensor, [batch_num, 3, sequence_length, batch_size, 42]
        batch_size: int, result batch size
    """
    assert isinstance(sequences, list), "input must be list, got {}".format(
        sequences.__class__.__name__)
    logger.info("Transfering tensors to batches, length {}, mode {}".format(
        len(sequences), mode))
    # Just convert it to tensors and all is done
    tensors = sequences_to_tensors(sequences)
    batches, batch_size = tensors_to_batches(tensors, mode, **kwargs)
    logger.info("Transfer completed, batch_num {}".format(len(batches)))
    return (batches, batch_size)


def getBatches(inputs, lead_only=False, mode='BATCH_NUM', **kwargs):
    """Lazyman API for both tensors and sequences
    This function requires an important `mode` parameter, which controls the batch process
    This fuction gurantee sequence_length are the same within a batch
    Args:
        inputs: list of torch.Tensor, [total_num, 3, sequence_length, 3]
                                   or [total_num, 3, sequence_length, 42]
        lead_only: bool, whether to output only the leading track
        mode: str, mode control
              'BATCH_NUM': DEFAULT, function will try to satisfy batch number
                           if necessary, some short sequences will be padded
                           when this mode selected, kwarg `batch_num` is availible
                           by default, batch_num will be 20
              'KEEP_LEN': function will try to keep sequence length,
                          if some length have only one sample, batch size will be 1
                          when this mode selected, kwarg `tolorance` is availible
                          by default, tolorance will be 0
    Returns:
        batches: list of torch.Tensor, [batch_num, 3, sequence_length, batch_size, 42]
                 if `lead_only == True`, will sized [batch_num, sequence_length, batch_size, 42]
        batch_size: int, result batch size
    """
    assert isinstance(inputs, list), "input must be list, got {}".format(
        inputs.__class__.__name__)
    clue = inputs[0].size()[-1]
    if clue == 3:
        logger.info("Lazy API called! inputs are sequences, calling ...")
        batches, batch_size = sequences_to_batches(inputs, mode, **kwargs)
    elif clue == 42:
        logger.info("Lazy API called! inputs are tensors, calling ...")
        batches, batch_size = tensors_to_batches(inputs, mode, **kwargs)
    else:
        logger.error("wrong input size, got {}".format(clue))
        exit()
    # select lead track if assigned
    if lead_only:
        logger.info("lead_only assigned! now returns only lead track")
        batches = [batch[0] for batch in batches]
    return (batches, batch_size)
