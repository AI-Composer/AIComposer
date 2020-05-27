"""Base Music Variational Autoencoder (MusicVAE) model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

import tensorflow.compat.v1 as tf
from tensorflow.contrib import metrics as contrib_metrics


class BaseEncoder(nn.Module):
    """Abstract encoder class.
    Implementations must define the following abstract methods:
     -`forward`
  """
    @abc.abstractmethod
    def __init__(self, hparams, output_depth, is_training=True):
        """Initialization for BaseEncoder.
    Args:
      hparams: An HParams object containing model hyperparameters.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model is being used for training.
    """
        super(BaseEncoder, self).__init__()
        pass

    @abc.abstractproperty
    def output_depth(self):
        """Returns the size of the output final dimension."""
        pass

    @abc.abstractmethod
    def encode(self, hparams, is_training=True):
        """Builder method for BaseEncoder.
    Args:
      hparams: An HParams object containing model hyperparameters.
      is_training: Whether or not the model is being used for training.
    """
        pass


class BaseDecoder(nn.Module):
    """Abstract decoder class.
  Implementations must define the following abstract methods:
     -`reconstruction_loss`
     -`sample`
  """
    @abc.abstractmethod
    def __init__(self, hparams, output_depth, is_training=True):
        """Initialization for BaseDecoder.
    Args:
      hparams: An HParams object containing model hyperparameters.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model is being used for training.
    """
        super(BaseEncoder, self).__init__()
        pass

    @abc.abstractmethod
    def reconstruction_loss(self,
                            x_input,
                            x_target,
                            x_length,
                            z=None,
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
          `[batch_size, max(x_length), control_depth]`. Required if
          conditioning on control sequences.
    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
    """
        pass

    @abc.abstractmethod
    def sample(self, n, max_length=None, z=None, c_input=None):
        """Sample from decoder with an optional conditional latent vector `z`.
    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
    """
        pass


class MusicVAE(object):
    """Music Variational Autoencoder."""
    def __init__(self, encoder, decoder, hparams, output_depth, is_training):
        """Initializer for a MusicVAE model.
    Args:
      encoder: A BaseEncoder implementation class to use.
      decoder: A BaseDecoder implementation class to use.
      hparams: An HParams object containing model hyperparameters. See
          `get_default_hparams` below for required values.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model will be used for training.
    """
        print('Initializing MusicVAE model with {}}, {}}, and hparams:\n{}'.
              format(self.encoder.__class__.__name__,
                     self.decoder.__class__.__name__, hparams.values()))
        self.global_step = hparams.global_step
        self._encoder = encoder
        self._decoder = decoder
        self._hparams = hparams
        self._output_depth = output_depth

        self.fc1 = nn.Linear(self.encoder.output_depth, hparams.z_size)
        self.fc2 = nn.Linear(self.encoder.output_depth, hparams.z_size)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def hparams(self):
        return self._hparams

    @property
    def output_depth(self):
        return self._output_depth

    def encode(self, sequence, sequence_length, control_sequence=None):
        """Encodes input sequences into a MultivariateNormalDiag distribution.
    Args:
      sequence: A Tensor with shape `[num_sequences, max_length, input_depth]`
          containing the sequences to encode.
      sequence_length: The length of each sequence in the `sequence` Tensor.
      control_sequence: (Optional) A Tensor with shape
          `[num_sequences, max_length, control_depth]` containing control
          sequences on which to condition. These will be concatenated depthwise
          to the input sequences.
    Returns:
      A torch.distributions.Normal representing the posterior
      distribution for each sequence.
    """
        sequence = sequence.to(torch.float32)
        if control_sequence is not None:
            control_sequence = control_sequence.to(torch.float32)
            sequence = torch.cat([sequence, control_sequence], dim=-1)
        encoder_output = self.encoder.encode(sequence, sequence_length)

        # Magenta verson used a fixed normal distribution for mu and sigma
        # For simpleness, I just use 2 trainable Linears for it
        """
        mu = tf.layers.dense(
            encoder_output,
            z_size,
            name='encoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))
        sigma = tf.layers.dense(
            encoder_output,
            z_size,
            activation=tf.nn.softplus,
            name='encoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))
        """
        mu = self.fc1(encoder_output)
        sigma = F.softplus(self.fc2(encoder_output))

        # Here another high level API used, I'll change it to a simple one
        """
        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        """
        # FIXME This line is likely to be wrong. WAITING FOR DEBUG
        return D.Normal(loc=mu, scale=sigma)

    def _compute_model_loss(self, input_sequence, output_sequence,
                            sequence_length, control_sequence):
        """Builds a model with loss for train/eval."""
        hparams = self.hparams
        batch_size = hparams.batch_size

        input_sequence = input_sequence.to(torch.float32)
        output_sequence = output_sequence.to(torch.float32)

        max_seq_len = torch.min(output_sequence.size()[1], hparams.max_seq_len)

        input_sequence = input_sequence[:, :max_seq_len]

        if control_sequence is not None:
            control_depth = control_sequence.shape[-1]
            control_sequence = control_sequence.to(torch.float32)
            control_sequence = control_sequence[:, :max_seq_len]
            # Shouldn't be necessary, but the slice loses shape information
            # when control depth is zero.
            control_sequence.view([batch_size, max_seq_len, control_depth])

        # The target/expected outputs.
        x_target = output_sequence[:, :max_seq_len]
        # Inputs fed to decoder, including zero padding for the initial input.
        x_input = F.pad(output_sequence[:, :max_seq_len - 1],
                        (0, 0, 1, 0, 0, 0))
        x_length = torch.min(sequence_length, max_seq_len)

        # Either encode to get `z`, or do unconditional, decoder-only.
        if hparams.z_size:  # vae mode:
            q_z = self.encode(input_sequence, x_length, control_sequence)
            z = q_z.sample()

            # Prior distribution.
            p_z = D.Normal(loc=[0.] * hparams.z_size,
                           scale=[1.] * hparams.z_size)

            # KL Divergence (nats)
            kl_div = D.kl_divergence(q_z, p_z)

            # Concatenate the Z vectors to the inputs at each time step.
        else:  # unconditional, decoder-only generation
            kl_div = torch.zeros([batch_size, 1], dtype=torch.float32)
            z = None

        r_loss, metric_map = self.decoder.reconstruction_loss(
            x_input, x_target, x_length, z, control_sequence)[0:2]

        free_nats = hparams.free_bits * torch.log(2.0)
        kl_cost = torch.max(kl_div - free_nats, 0)

        # FIXME WTF is this ?!
        beta = ((1.0 - torch.pow(hparams.beta_rate,
                                 self.global_step.to(torch.float32))) *
                hparams.max_beta)
        self.loss = torch.mean(r_loss) + beta * torch.mean(kl_cost)

        scalars_to_summarize = {
            'loss': self.loss,
            'losses/r_loss': r_loss,
            'losses/kl_loss': kl_cost,
            'losses/kl_bits': kl_div / torch.log(2.0),
            'losses/kl_beta': beta,
        }
        return metric_map, scalars_to_summarize

    def train(self,
              input_sequence,
              output_sequence,
              sequence_length,
              control_sequence=None):
        """Train on the given sequences, returning an optimizer.
    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
          identical).
      control_sequence: (Optional) sequence on which to condition. This will be
          concatenated depthwise to the model inputs for both encoding and
          decoding.
    Returns:
      optimizer: A torch.optim.optimizer.
    """

        _, scalars_to_summarize = self._compute_model_loss(
            input_sequence, output_sequence, sequence_length, control_sequence)

        hparams = self.hparams
        lr = (
            (hparams.learning_rate - hparams.min_learning_rate) *
            torch.pow(hparams.decay_rate, self.global_step.to(torch.float32)) +
            hparams.min_learning_rate)

        optimizer = optim.Adam(lr=lr)

        print('learning_rate', lr)
        for n, t in scalars_to_summarize.items():
            print(n, torch.mean(t))

        return optimizer

    # TODO
    def eval(self,
             input_sequence,
             output_sequence,
             sequence_length,
             control_sequence=None):
        """Evaluate on the given sequences, returning metric update ops.
    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
        identical).
      control_sequence: (Optional) sequence on which to condition the decoder.
    Returns:
      metric_update_ops: tf.metrics update ops.
    """
        metric_map, scalars_to_summarize = self._compute_model_loss(
            input_sequence, output_sequence, sequence_length, control_sequence)

        for n, t in scalars_to_summarize.items():
            metric_map[n] = torch.mean(t)

        metrics_to_values, metrics_to_updates = (
            contrib_metrics.aggregate_metric_map(metric_map))

        for metric_name, metric_value in metrics_to_values.items():
            tf.summary.scalar(metric_name, metric_value)

        return list(metrics_to_updates.values())

    def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
        """Sample with an optional conditional embedding `z`."""
        if z is not None and z.shape[0].value != n:
            raise ValueError(
                '`z` must have a first dimension that equals `n` when given. '
                'Got: %d vs %d' % (z.shape[0].value, n))

        if self.hparams.z_size and z is None:
            print(
                'Sampling from conditional model without `z`. Using random `z`.'
            )
            normal_shape = [n, self.hparams.z_size]
            normal_dist = D.Normal(loc=torch.zeros(normal_shape),
                                   scale=torch.ones(normal_shape))
            z = normal_dist.sample()

        return self.decoder.sample(n, max_length, z, c_input, **kwargs)


class hparams:
    """
    If we want to use magenta structure,
    addtional classes in tensorflow must be manually defined
    """
    def __init__(
            self,
            max_seq_len=32,
            z_size=32,
            free_bits=0.0,
            max_beta=1.0,
            beta_rate=0.0,pytorch重复tensor
            batch_size=512,
            grad_clip=1.0,
            clip_mode='global_norm',
            grad_norm_clip_to_zero=10000,
            learning_rate=0.001,
            decay_rate=0.9999,
            min_learning_rate=0.00001,
    ):
        self.max_seq_len = max_seq_len  # Maximum sequence length. Others will be truncated.
        self.z_size = z_size  # Size of latent vector z.
        self.free_bits = 0.0  # Bits to exclude from KL loss per dimension.
        self.max_beta = max_beta  # Maximum KL cost weight, or cost if not annealing.
        self.beta_rate = beta_rate  # Exponential rate at which to anneal KL cost.
        self.batch_size = batch_size  # Minibatch size.
        self.grad_clip = grad_clip  # Gradient clipping. Recommend leaving at 1.0.
        self.clip_mode = 'global_norm'  # value or global_norm.
        # If clip_mode=global_norm and global_norm is greater than this value,
        # the gradient will be clipped to 0, effectively ignoring the step.
        self.grad_norm_clip_to_zero = grad_norm_clip_to_zero
        self.learning_rate = learning_rate  # Learning rate.
        self.decay_rate = decay_rate  # Learning rate decay per minibatch.
        self.min_learning_rate = min_learning_rate  # Minimum learning rate.


def get_default_hparams():
    return hparams()
