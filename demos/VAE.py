import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds


class VAENet(nn.Module):
    """A really simplified demo of VAE net"""
    def __init__(self,
                 input_depth,
                 hidden_size=200,
                 num_layers=1,
                 z_size=32,
                 free_bits=0,
                 beta_rate=0,
                 max_beta=1,
                 lr=0.001):
        self.z_size = z_size
        self.free_bits = free_bits
        self.beta_rate = beta_rate
        self.max_beta = max_beta

        self.step = 0

        self.bidlstm1 = nn.LSTM(input_depth,
                                hidden_size,
                                num_layers,
                                bidirectional=True)
        self.bidlstm2 = nn.LSTM(input_depth,
                                hidden_size,
                                num_layers,
                                bidirectional=True)
        self.bidlstm3 = nn.LSTM(input_depth,
                                hidden_size,
                                num_layers,
                                bidirectional=True)
        self.fc1 = nn.Linear(3 * 2 * hidden_size, z_size)
        self.fc2 = nn.Linear(3 * 2 * hidden_size, z_size)
        self.fc3 = nn.Linear(z_size, hidden_size)
        self.lstm1 = nn.LSTM(input_depth, hidden_size, num_layers)
        self.lstm2 = nn.LSTM(input_depth, hidden_size, num_layers)
        self.lstm3 = nn.LSTM(input_depth, hidden_size, num_layers)
        self.fc4 = nn.Linear(hidden_size, input_depth)
        self.fc5 = nn.Linear(hidden_size, input_depth)
        self.fc6 = nn.Linear(hidden_size, input_depth)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def encode(self, x):
        """encode process demo
        Args:
            x: A batch of sequences [sequence_length, batch_size, input_depth, 3]
        Returns:
            distribution: distribution of z torch.distributions.Normal
        """
        # FIXME I'm not sure whether to use hidden state or cell state
        _, (hidden_state1, _) = self.bidlstm1(x[:, :, :, 0])
        _, (hidden_state2, _) = self.bidlstm2(x[:, :, :, 1])
        _, (hidden_state3, _) = self.bidlstm3(x[:, :, :, 2])
        # hidden state [2, batch_size, hidden_size]
        hidden_state = torch.cat((hidden_state1, hidden_state2, hidden_state3),
                                 dim=2)
        # hidden state [2, batch_size, 3*hidden_size]
        hidden_state = torch.flatten(
            torch.transpose(hidden_state, dim0=0, dim1=1))
        # hidden state [batch_size, 3*2*hidden_size]
        # compute mu and sigma
        mu = self.fc1(hidden_state)
        sigma = F.softplus(self.fc2(hidden_state))
        # form distribution
        distribution = ds.Normal(loc=mu, scale=sigma)
        return distribution

    def forward(self, x):
        """Forward through VAE net
        Args:
            x: A batch of sequences [sequence_length, batch_size, input_depth, 3]
        Returns:
            distribution: distribution of z torch.distributions.Normal
            output: [sequence_length, batch_size, input_depth, 3]
        """
        distribution = self.encode(x)
        # sample from the distribution
        z = distribution.sample()
        # z [batch_size]
        # decode
        output = self.fc3(z)
        state1, (_, _) = self.lstm1(output)
        state2, (_, _) = self.lstm2(output)
        state3, (_, _) = self.lstm3(output)
        # state1/state2/state3 [sequence_length, batch_size, hidden_size]
        output1 = self.fc4(state1)
        output2 = self.fc5(state2)
        output3 = self.fc6(state3)
        # output1/output2/output3 [sequence_length, batch_size, input_depth]
        return (distribution, torch.stack((output1, output2, output3), dim=3))

    def generate(self, x, num_sequences):
        """generate new music demo
        Args:
            inputs: [sequence_length, batch_size, input_depth, 3]
        Returns:
            outputs: [num_sequences*sequence_length, batch_size, input_depth, 3]
        """
        outputs = None
        for _ in range(num_sequences):
            distribution = self.encode(x)
            # sample from the distribution
            z = distribution.sample()
            # z [batch_size]
            # decode
            output = self.fc3(z)
            state1, (_, _) = self.lstm1(output)
            state2, (_, _) = self.lstm2(output)
            state3, (_, _) = self.lstm3(output)
            # state1/state2/state3 [sequence_length, batch_size, hidden_size]
            output1 = self.fc4(state1)
            output2 = self.fc5(state2)
            output3 = self.fc6(state3)
            # output1/output2/output3 [sequence_length, batch_size, input_depth]
            # update x
            x = torch.stack((output1, output2, output3), dim=3)
            # concat new paragraph to outputs
            if outputs is not None:
                torch.cat((outputs, x), dim=0)
            else:
                outputs = x
        return outputs

    def loss_fn(self, inputs, outputs, distribution):
        """music VAE loss function, complicated, mainly KL_divergence
        Args:
            inputs: [sequence_length, batch_size, input_depth, 3]
            outputs: [sequence_length, batch_size, input_depth, 3]
            distribution: torch.distributions.Normal
        Returns:
            loss: scalar
        """
        q_z = distribution
        p_z = ds.Normal(loc=[0.] * self.z_size, scale=[1.] * self.z_size)
        kl_div = ds.kl_divergence(q_z, p_z)
        # allow some free bits
        free_nats = self.free_bits * torch.log(2.0)
        # this cost limit the encoder to approach a standard distribution for z
        kl_cost = torch.mean(torch.max(kl_div - free_nats, 0))

        # musicVAE decoder loss function is too complicated, simplified here
        # FIXME This function can be really unreasonable !!!
        r_cost = torch.mean(F.mse_loss(inputs, outputs))

        # Now use a proper weight to balance between the two losses
        beta = (
            (1.0 - torch.pow(self.beta_rate, self.step.to(torch.float32))) *
            self.max_beta)
        loss = torch.mean(r_cost) + beta * torch.mean(kl_cost)
        return loss

    def train(self, inputs):
        """training process demo
        Args:
            inputs: [sequence_length, batch_size, input_depth, 3]
        Returns:
            None
        """
        self.train()

        distribution, outputs = self.forward(inputs)
        loss = self.loss_fn(inputs, outputs, distribution)
        loss.backward()

        self.optimizer.zero_grad()

        self.optimizer.step()

        self.step += 1
