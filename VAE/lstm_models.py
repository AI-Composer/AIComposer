"""LSTM-based encoders and decoders for MusicVAE."""
import abc

from magenta.common import flatten_maybe_padded_sequences
from magenta.common import Nade
from magenta.models.music_vae import base_model
from magenta.models.music_vae import lstm_utils
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib import seq2seq as contrib_seq2seq
from tensorflow.contrib import training as contrib_training

rnn = contrib_rnn
seq2seq = contrib_seq2seq

# ENCODERS


class LstmEncoder(base_model.BaseEncoder):
  """Unidirectional LSTM Encoder."""

  @property
  def output_depth(self):
    return self._cell.output_size

  def build(self, hparams, is_training=True, name_or_scope='encoder'):
    if hparams.use_cudnn and hparams.residual_encoder:
      raise ValueError('Residual connections not supported in cuDNN.')

    self._is_training = is_training
    self._name_or_scope = name_or_scope
    self._use_cudnn = hparams.use_cudnn

    tf.logging.info('\nEncoder Cells (unidirectional):\n'
                    '  units: %s\n',
                    hparams.enc_rnn_size)
    if self._use_cudnn:
      self._cudnn_lstm = lstm_utils.cudnn_lstm_layer(
          hparams.enc_rnn_size,
          hparams.dropout_keep_prob,
          is_training,
          name_or_scope=self._name_or_scope)
    else:
      self._cell = lstm_utils.rnn_cell(
          hparams.enc_rnn_size, hparams.dropout_keep_prob,
          hparams.residual_encoder, is_training)

  def encode(self, sequence, sequence_length):
    # Convert to time-major.
    sequence = tf.transpose(sequence, [1, 0, 2])
    if self._use_cudnn:
      outputs, _ = self._cudnn_lstm(
          sequence, training=self._is_training)
      return lstm_utils.get_final(outputs, sequence_length)
    else:
      outputs, _ = tf.nn.dynamic_rnn(
          self._cell, sequence, sequence_length, dtype=tf.float32,
          time_major=True, scope=self._name_or_scope)
      return outputs[-1]


class BidirectionalLstmEncoder(base_model.BaseEncoder):
  """Bidirectional LSTM Encoder."""

  @property
  def output_depth(self):
    if self._use_cudnn:
      return self._cells[0][-1].num_units + self._cells[1][-1].num_units
    return self._cells[0][-1].output_size + self._cells[1][-1].output_size

  def build(self, hparams, is_training=True, name_or_scope='encoder'):
    self._is_training = is_training
    self._name_or_scope = name_or_scope
    self._use_cudnn = hparams.use_cudnn

    tf.logging.info('\nEncoder Cells (bidirectional):\n'
                    '  units: %s\n',
                    hparams.enc_rnn_size)

    self._cells = lstm_utils.build_bidirectional_lstm(
        layer_sizes=hparams.enc_rnn_size,
        use_cudnn=self._use_cudnn,
        dropout_keep_prob=hparams.dropout_keep_prob,
        residual=hparams.residual_encoder,
        is_training=is_training,
        name_or_scope=name_or_scope)

  def encode(self, sequence, sequence_length):
    cells_fw, cells_bw = self._cells

    if self._use_cudnn:
      outputs_fw, outputs_bw = lstm_utils.cudnn_bidirectional_lstm(
          cells_fw, cells_bw, sequence, sequence_length, self._is_training)
      last_h_fw = lstm_utils.get_final(outputs_fw, sequence_length)
      # outputs_bw has already been reversed, so we can take the first element.
      last_h_bw = outputs_bw[0]

    else:
      _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
          cells_fw,
          cells_bw,
          sequence,
          sequence_length=sequence_length,
          time_major=False,
          dtype=tf.float32,
          scope=self._name_or_scope)
      # Note we access the outputs (h) from the states since the backward
      # ouputs are reversed to the input order in the returned outputs.
      last_h_fw = states_fw[-1][-1].h
      last_h_bw = states_bw[-1][-1].h

    return tf.concat([last_h_fw, last_h_bw], 1)


class HierarchicalLstmEncoder(base_model.BaseEncoder):
  """Hierarchical LSTM encoder wrapper.
  Input sequences will be split into segments based on the first value of
  `level_lengths` and encoded. At subsequent levels, the embeddings will be
  grouped based on `level_lengths` and encoded until a single embedding is
  produced.
  See the `encode` method for details on the expected arrangement the sequence
  tensors.
  Args:
    core_encoder_cls: A single BaseEncoder class to use for each level of the
      hierarchy.
    level_lengths: A list of the (maximum) lengths of the segments at each
      level of the hierarchy. The product must equal `hparams.max_seq_len`.
  """

  def __init__(self, core_encoder_cls, level_lengths):
    self._core_encoder_cls = core_encoder_cls
    self._level_lengths = level_lengths

  @property
  def output_depth(self):
    return self._hierarchical_encoders[-1][1].output_depth

  @property
  def level_lengths(self):
    return list(self._level_lengths)

  def level(self, l):
    """Returns the BaseEncoder at level `l`."""
    return self._hierarchical_encoders[l][1]

  def build(self, hparams, is_training=True):
    self._total_length = hparams.max_seq_len
    if self._total_length != np.prod(self._level_lengths):
      raise ValueError(
          'The product of the HierarchicalLstmEncoder level lengths (%d) must '
          'equal the padded input sequence length (%d).' % (
              np.prod(self._level_lengths), self._total_length))
    tf.logging.info('\nHierarchical Encoder:\n'
                    '  input length: %d\n'
                    '  level lengths: %s\n',
                    self._total_length,
                    self._level_lengths)
    self._hierarchical_encoders = []
    num_splits = np.prod(self._level_lengths)
    for i, l in enumerate(self._level_lengths):
      num_splits //= l
      tf.logging.info('Level %d splits: %d', i, num_splits)
      h_encoder = self._core_encoder_cls()
      h_encoder.build(
          hparams, is_training,
          name_or_scope=tf.VariableScope(
              tf.AUTO_REUSE, 'encoder/hierarchical_level_%d' % i))
      self._hierarchical_encoders.append((num_splits, h_encoder))

  def encode(self, sequence, sequence_length):
    """Hierarchically encodes the input sequences, returning a single embedding.
    Each sequence should be padded per-segment. For example, a sequence with
    three segments [1, 2, 3], [4, 5], [6, 7, 8 ,9] and a `max_seq_len` of 12
    should be input as `sequence = [1, 2, 3, 0, 4, 5, 0, 0, 6, 7, 8, 9]` with
    `sequence_length = [3, 2, 4]`.
    Args:
      sequence: A batch of (padded) sequences, sized
        `[batch_size, max_seq_len, input_depth]`.
      sequence_length: A batch of sequence lengths. May be sized
        `[batch_size, level_lengths[0]]` or `[batch_size]`. If the latter,
        each length must either equal `max_seq_len` or 0. In this case, the
        segment lengths are assumed to be constant and the total length will be
        evenly divided amongst the segments.
    Returns:
      embedding: A batch of embeddings, sized `[batch_size, N]`.
    """
    batch_size = sequence.shape[0].value
    sequence_length = lstm_utils.maybe_split_sequence_lengths(
        sequence_length, np.prod(self._level_lengths[1:]),
        self._total_length)

    for level, (num_splits, h_encoder) in enumerate(
        self._hierarchical_encoders):
      split_seqs = tf.split(sequence, num_splits, axis=1)
      # In the first level, we use the input `sequence_length`. After that,
      # we use the full embedding sequences.
      if level:
        sequence_length = tf.fill(
            [batch_size, num_splits], split_seqs[0].shape[1])
      split_lengths = tf.unstack(sequence_length, axis=1)
      embeddings = [
          h_encoder.encode(s, l) for s, l in zip(split_seqs, split_lengths)]
      sequence = tf.stack(embeddings, axis=1)

    with tf.control_dependencies([tf.assert_equal(tf.shape(sequence)[1], 1)]):
      return sequence[:, 0]


# DECODERS


class BaseLstmDecoder(base_model.BaseDecoder):
  """Abstract LSTM Decoder class.
  Implementations must define the following abstract methods:
      -`_sample`
      -`_flat_reconstruction_loss`
  """

  def build(self, hparams, output_depth, is_training=True):
    if hparams.use_cudnn and hparams.residual_decoder:
      raise ValueError('Residual connections not supported in cuDNN.')

    self._is_training = is_training

    tf.logging.info('\nDecoder Cells:\n'
                    '  units: %s\n',
                    hparams.dec_rnn_size)

    self._sampling_probability = lstm_utils.get_sampling_probability(
        hparams, is_training)
    self._output_depth = output_depth
    self._output_layer = tf.layers.Dense(
        output_depth, name='output_projection')
    self._dec_cell = lstm_utils.rnn_cell(
        hparams.dec_rnn_size, hparams.dropout_keep_prob,
        hparams.residual_decoder, is_training)
    if hparams.use_cudnn:
      self._cudnn_dec_lstm = lstm_utils.cudnn_lstm_layer(
          hparams.dec_rnn_size, hparams.dropout_keep_prob, is_training,
          name_or_scope='decoder')
    else:
      self._cudnn_dec_lstm = None

  @property
  def state_size(self):
    return self._dec_cell.state_size

  @abc.abstractmethod
  def _sample(self, rnn_output, temperature):
    """Core sampling method for a single time step.
    Args:
      rnn_output: The output from a single timestep of the RNN, sized
          `[batch_size, rnn_output_size]`.
      temperature: A scalar float specifying a sampling temperature.
    Returns:
      A batch of samples from the model.
    """
    pass

  @abc.abstractmethod
  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    """Core loss calculation method for flattened outputs.
    Args:
      flat_x_target: The flattened ground truth vectors, sized
        `[sum(x_length), self._output_depth]`.
      flat_rnn_output: The flattened output from all timeputs of the RNN,
        sized `[sum(x_length), rnn_output_size]`.
    Returns:
      r_loss: The unreduced reconstruction losses, sized `[sum(x_length)]`.
      metric_map: A map of metric names to tuples, each of which contain the
        pair of (value_tensor, update_op) from a tf.metrics streaming metric.
    """
    pass

  def _decode(self, z, helper, input_shape, max_length=None):
    """Decodes the given batch of latent vectors vectors, which may be 0-length.
    Args:
      z: Batch of latent vectors, sized `[batch_size, z_size]`, where `z_size`
        may be 0 for unconditioned decoding.
      helper: A seq2seq.Helper to use. If a TrainingHelper is passed and a
        CudnnLSTM has previously been defined, it will be used instead.
      input_shape: The shape of each model input vector passed to the decoder.
      max_length: (Optional) The maximum iterations to decode.
    Returns:
      results: The LstmDecodeResults.
    """
    initial_state = lstm_utils.initial_cell_state_from_embedding(
        self._dec_cell, z, name='decoder/z_to_initial_state')

    # CudnnLSTM does not support sampling so it can only replace TrainingHelper.
    if  self._cudnn_dec_lstm and type(helper) is seq2seq.TrainingHelper:  # pylint:disable=unidiomatic-typecheck
      rnn_output, _ = self._cudnn_dec_lstm(
          tf.transpose(helper.inputs, [1, 0, 2]),
          initial_state=lstm_utils.state_tuples_to_cudnn_lstm_state(
              initial_state),
          training=self._is_training)
      with tf.variable_scope('decoder'):
        rnn_output = self._output_layer(rnn_output)

      results = lstm_utils.LstmDecodeResults(
          rnn_input=helper.inputs[:, :, :self._output_depth],
          rnn_output=tf.transpose(rnn_output, [1, 0, 2]),
          samples=tf.zeros([z.shape[0], 0]),
          # TODO(adarob): Pass the final state when it is valid (fixed-length).
          final_state=None,
          final_sequence_lengths=helper.sequence_length)
    else:
      if self._cudnn_dec_lstm:
        tf.logging.warning(
            'CudnnLSTM does not support sampling. Using `dynamic_decode` '
            'instead.')
      decoder = lstm_utils.Seq2SeqLstmDecoder(
          self._dec_cell,
          helper,
          initial_state=initial_state,
          input_shape=input_shape,
          output_layer=self._output_layer)
      final_output, final_state, final_lengths = seq2seq.dynamic_decode(
          decoder,
          maximum_iterations=max_length,
          swap_memory=True,
          scope='decoder')
      results = lstm_utils.LstmDecodeResults(
          rnn_input=final_output.rnn_input[:, :, :self._output_depth],
          rnn_output=final_output.rnn_output,
          samples=final_output.sample_id,
          final_state=final_state,
          final_sequence_lengths=final_lengths)

    return results

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.
    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
        `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
        sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
        `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
          `[batch_size, max(x_length), control_depth]`. Required if conditioning
          on control sequences.
    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      decode_results: The LstmDecodeResults.
    """
    batch_size = x_input.shape[0].value

    has_z = z is not None
    z = tf.zeros([batch_size, 0]) if z is None else z
    repeated_z = tf.tile(
        tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])

    has_control = c_input is not None
    if c_input is None:
      c_input = tf.zeros([batch_size, tf.shape(x_input)[1], 0])

    sampling_probability_static = tf.get_static_value(
        self._sampling_probability)
    if sampling_probability_static == 0.0:
      # Use teacher forcing.
      x_input = tf.concat([x_input, repeated_z, c_input], axis=2)
      helper = seq2seq.TrainingHelper(x_input, x_length)
    else:
      # Use scheduled sampling.
      if has_z or has_control:
        auxiliary_inputs = tf.zeros([batch_size, tf.shape(x_input)[1], 0])
        if has_z:
          auxiliary_inputs = tf.concat([auxiliary_inputs, repeated_z], axis=2)
        if has_control:
          auxiliary_inputs = tf.concat([auxiliary_inputs, c_input], axis=2)
      else:
        auxiliary_inputs = None
      helper = seq2seq.ScheduledOutputTrainingHelper(
          inputs=x_input,
          sequence_length=x_length,
          auxiliary_inputs=auxiliary_inputs,
          sampling_probability=self._sampling_probability,
          next_inputs_fn=self._sample)

    decode_results = self._decode(
        z, helper=helper, input_shape=helper.inputs.shape[2:])
    flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
    flat_rnn_output = flatten_maybe_padded_sequences(
        decode_results.rnn_output, x_length)
    r_loss, metric_map = self._flat_reconstruction_loss(
        flat_x_target, flat_rnn_output)

    # Sum loss over sequences.
    cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
    r_losses = []
    for i in range(batch_size):
      b, e = cum_x_len[i], cum_x_len[i + 1]
      r_losses.append(tf.reduce_sum(r_loss[b:e]))
    r_loss = tf.stack(r_losses)

    return r_loss, metric_map, decode_results

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
             start_inputs=None, end_fn=None):
    """Sample from decoder with an optional conditional latent vector `z`.
    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      temperature: (Optional) The softmax temperature to use when sampling, if
        applicable.
      start_inputs: (Optional) Initial inputs to use for batch.
        Sized `[n, output_depth]`.
      end_fn: (Optional) A callable that takes a batch of samples (sized
        `[n, output_depth]` and emits a `bool` vector
        shaped `[batch_size]` indicating whether each sample is an end token.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
      final_state: The final states of the decoder.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`.
    """
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    # Use a dummy Z in unconditional case.
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    if c_input is not None:
      # Tile control sequence across samples.
      c_input = tf.tile(tf.expand_dims(c_input, 1), [1, n, 1])

    # If not given, start with zeros.
    if start_inputs is None:
      start_inputs = tf.zeros([n, self._output_depth], dtype=tf.float32)
    # In the conditional case, also concatenate the Z.
    start_inputs = tf.concat([start_inputs, z], axis=-1)
    if c_input is not None:
      start_inputs = tf.concat([start_inputs, c_input[0]], axis=-1)
    initialize_fn = lambda: (tf.zeros([n], tf.bool), start_inputs)

    sample_fn = lambda time, outputs, state: self._sample(outputs, temperature)
    end_fn = end_fn or (lambda x: False)

    def next_inputs_fn(time, outputs, state, sample_ids):
      del outputs
      finished = end_fn(sample_ids)
      next_inputs = tf.concat([sample_ids, z], axis=-1)
      if c_input is not None:
        # We need to stop if we've run out of control input.
        finished = tf.cond(tf.less(time, tf.shape(c_input)[0] - 1),
                           lambda: finished,
                           lambda: True)
        next_inputs = tf.concat([
            next_inputs,
            tf.cond(tf.less(time, tf.shape(c_input)[0] - 1),
                    lambda: c_input[time + 1],
                    lambda: tf.zeros_like(c_input[0]))  # should be unused
        ], axis=-1)
      return (finished, next_inputs, state)

    sampler = seq2seq.CustomHelper(
        initialize_fn=initialize_fn, sample_fn=sample_fn,
        next_inputs_fn=next_inputs_fn, sample_ids_shape=[self._output_depth],
        sample_ids_dtype=tf.float32)

    decode_results = self._decode(
        z, helper=sampler, input_shape=start_inputs.shape[1:],
        max_length=max_length)

    return decode_results.samples, decode_results


class BidirectionalLstmControlPreprocessingDecoder(base_model.BaseDecoder):
  """Decoder that preprocesses control input with a bidirectional LSTM."""

  def __init__(self, core_decoder):
    super(BidirectionalLstmControlPreprocessingDecoder, self).__init__()
    self._core_decoder = core_decoder

  def build(self, hparams, output_depth, is_training=True):
    self._is_training = is_training
    self._use_cudnn = hparams.use_cudnn

    tf.logging.info('\nControl Preprocessing Cells (bidirectional):\n'
                    '  units: %s\n',
                    hparams.control_preprocessing_rnn_size)

    self._control_preprocessing_cells = lstm_utils.build_bidirectional_lstm(
        layer_sizes=hparams.control_preprocessing_rnn_size,
        use_cudnn=self._use_cudnn,
        dropout_keep_prob=hparams.dropout_keep_prob,
        residual=hparams.residual_decoder,
        is_training=is_training,
        name_or_scope='control_preprocessing')

    self._core_decoder.build(hparams, output_depth, is_training)

  def _preprocess_controls(self, c_input, length):
    cells_fw, cells_bw = self._control_preprocessing_cells

    if self._use_cudnn:
      outputs_fw, outputs_bw = lstm_utils.cudnn_bidirectional_lstm(
          cells_fw, cells_bw, c_input, length, self._is_training)
      outputs = tf.transpose(
          tf.concat([outputs_fw, outputs_bw], axis=2), [1, 0, 2])

    else:
      outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
          cells_fw,
          cells_bw,
          c_input,
          sequence_length=length,
          time_major=False,
          dtype=tf.float32,
          scope='control_preprocessing')

    return outputs

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    if c_input is None:
      raise ValueError('Must provide control input to preprocess.')
    preprocessed_c_input = self._preprocess_controls(c_input, x_length)
    return self._core_decoder.reconstruction_loss(
        x_input, x_target, x_length, z, preprocessed_c_input)

  def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
    if c_input is None:
      raise ValueError('Must provide control input to preprocess.')
    preprocessed_c_input = tf.squeeze(
        self._preprocess_controls(tf.expand_dims(c_input, axis=0),
                                  tf.reshape(max_length, [1])),
        axis=0)
    return self._core_decoder.sample(
        n, max_length, z, preprocessed_c_input, **kwargs)


class BooleanLstmDecoder(BaseLstmDecoder):
  """LSTM decoder with single Boolean output per time step."""

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    flat_logits = flat_rnn_output
    flat_truth = tf.squeeze(flat_x_target, axis=1)
    flat_predictions = tf.squeeze(flat_logits >= 0, axis=1)
    r_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=flat_x_target, logits=flat_logits)

    metric_map = {
        'metrics/accuracy':
            tf.metrics.accuracy(flat_truth, flat_predictions),
    }
    return r_loss, metric_map

  def _sample(self, rnn_output, temperature=1.0):
    sampler = tfp.distributions.Bernoulli(
        logits=rnn_output / temperature, dtype=tf.float32)
    return sampler.sample()


class CategoricalLstmDecoder(BaseLstmDecoder):
  """LSTM decoder with single categorical output."""

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    flat_logits = flat_rnn_output
    flat_truth = tf.argmax(flat_x_target, axis=1)
    flat_predictions = tf.argmax(flat_logits, axis=1)
    r_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=flat_x_target, logits=flat_logits)

    metric_map = {
        'metrics/accuracy':
            tf.metrics.accuracy(flat_truth, flat_predictions),
        'metrics/mean_per_class_accuracy':
            tf.metrics.mean_per_class_accuracy(
                flat_truth, flat_predictions, flat_x_target.shape[-1].value),
    }
    return r_loss, metric_map

  def _sample(self, rnn_output, temperature=1.0):
    sampler = tfp.distributions.OneHotCategorical(
        logits=rnn_output / temperature, dtype=tf.float32)
    return sampler.sample()

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=None,
             start_inputs=None, beam_width=None, end_token=None):
    """Overrides BaseLstmDecoder `sample` method to add optional beam search.
    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      temperature: (Optional) The softmax temperature to use when not doing beam
        search. Defaults to 1.0. Ignored when `beam_width` is provided.
      start_inputs: (Optional) Initial inputs to use for batch.
        Sized `[n, output_depth]`.
      beam_width: (Optional) Width of beam to use for beam search. Beam search
        is disabled if not provided.
      end_token: (Optional) Scalar token signaling the end of the sequence to
        use for early stopping.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
      final_state: The final states of the decoder.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`,
        or if `c_input` is provided under beam search.
    """
    if beam_width is None:
      if end_token is None:
        end_fn = None
      else:
        end_fn = lambda x: tf.equal(tf.argmax(x, axis=-1), end_token)
      return super(CategoricalLstmDecoder, self).sample(
          n, max_length, z, c_input, temperature, start_inputs, end_fn)

    # TODO(iansimon): Support conditioning in beam search decoder, which may be
    # awkward as there's no helper.
    if c_input is not None:
      raise ValueError('Control sequence unsupported in beam search.')

    # If `end_token` is not given, use an impossible value.
    end_token = self._output_depth if end_token is None else end_token
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    if temperature is not None:
      tf.logging.warning('`temperature` is ignored when using beam search.')
    # Use a dummy Z in unconditional case.
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    # If not given, start with dummy `-1` token and replace with zero vectors in
    # `embedding_fn`.
    if start_inputs is None:
      start_tokens = -1 * tf.ones([n], dtype=tf.int32)
    else:
      start_tokens = tf.argmax(start_inputs, axis=-1, output_type=tf.int32)

    initial_state = lstm_utils.initial_cell_state_from_embedding(
        self._dec_cell, z, name='decoder/z_to_initial_state')
    beam_initial_state = seq2seq.tile_batch(
        initial_state, multiplier=beam_width)

    # Tile `z` across beams.
    beam_z = tf.tile(tf.expand_dims(z, 1), [1, beam_width, 1])

    def embedding_fn(tokens):
      # If tokens are the start_tokens (negative), replace with zero vectors.
      next_inputs = tf.cond(
          tf.less(tokens[0, 0], 0),
          lambda: tf.zeros([n, beam_width, self._output_depth]),
          lambda: tf.one_hot(tokens, self._output_depth))

      # Concatenate `z` to next inputs.
      next_inputs = tf.concat([next_inputs, beam_z], axis=-1)
      return next_inputs

    decoder = seq2seq.BeamSearchDecoder(
        self._dec_cell,
        embedding_fn,
        start_tokens,
        end_token,
        beam_initial_state,
        beam_width,
        output_layer=self._output_layer,
        length_penalty_weight=0.0)

    final_output, final_state, final_lengths = seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=max_length,
        swap_memory=True,
        scope='decoder')

    samples = tf.one_hot(final_output.predicted_ids[:, :, 0],
                         self._output_depth)
    # Rebuild the input by combining the inital input with the sampled output.
    if start_inputs is None:
      initial_inputs = tf.zeros([n, 1, self._output_depth])
    else:
      initial_inputs = tf.expand_dims(start_inputs, axis=1)

    rnn_input = tf.concat([initial_inputs, samples[:, :-1]