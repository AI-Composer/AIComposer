import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds


class VAENet(nn.Module):
    """A really simplified demo of VAE net"""
    def __init__(self,
                 input_depth,
                 section_length=48,
                 encoder_size=2048,
                 encoder_layers=1,
                 decoder_size=1024,
                 decoder_layers=2,
                 z_size=512,
                 conductor_size=1024,
                 control_depth=512,
                 free_bits=0,
                 beta_rate=0,
                 max_beta=1,
                 lr=0.001,
                 repeat=32):
        super(VAENet, self).__init__()
        self.input_depth = input_depth
        self.section_length = section_length
        self.z_size = z_size
        self.free_bits = free_bits
        self.beta_rate = beta_rate
        self.max_beta = max_beta
        self.repeat = repeat

        self.step = 0

        # Use three LSTM to independently grab latent code information
        self.bidlstm1 = nn.LSTM(input_depth,
                                encoder_size,
                                encoder_layers,
                                bidirectional=True)
        self.bidlstm2 = nn.LSTM(input_depth,
                                encoder_size,
                                encoder_layers,
                                bidirectional=True)
        self.bidlstm3 = nn.LSTM(input_depth,
                                encoder_size,
                                encoder_layers,
                                bidirectional=True)

        # Use 3*2 Dense to independently grab latent code
        # We don't use 3*2*hidden_size because we'll start from 1 track in the future
        self.mu1 = nn.Linear(2 * encoder_size, z_size)
        self.mu2 = nn.Linear(2 * encoder_size, z_size)
        self.mu3 = nn.Linear(2 * encoder_size, z_size)
        self.sigma1 = nn.Linear(2 * encoder_size, z_size)
        self.sigma2 = nn.Linear(2 * encoder_size, z_size)
        self.sigma3 = nn.Linear(2 * encoder_size, z_size)
        # Conductor
        self.fc1 = nn.Linear(z_size, control_depth)
        self.conductor = nn.LSTM(control_depth, conductor_size)
        self.fc2 = nn.Linear(conductor_size, control_depth)
        # LSTM decoder
        self.lstm1 = nn.LSTM(input_depth + control_depth, decoder_size,
                             decoder_layers)
        self.lstm2 = nn.LSTM(input_depth + control_depth, decoder_size,
                             decoder_layers)
        self.lstm3 = nn.LSTM(input_depth + control_depth, decoder_size,
                             decoder_layers)
        self.fc4 = nn.Linear(decoder_size, input_depth)
        self.fc5 = nn.Linear(decoder_size, input_depth)
        self.fc6 = nn.Linear(decoder_size, input_depth)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def compose(self, x, track=0):
        """encode process demo, 1 track input
        Args:
            x: A batch of sequences [sequence_length, batch_size, input_depth]
        Returns:
            mu: mu of Normal distribution
            sigma: sigma of Normal distribution
        """
        # Use not simplified code at all, for clearness
        if track == 0:
            _, (hidden_state1, _) = self.bidlstm1(x)
            hidden_state1 = torch.flatten(torch.transpose(hidden_state1,
                                                          dim0=0,
                                                          dim1=1),
                                          start_dim=1)
            mu = self.mu1(hidden_state1)
            sigma = self.sigma1(hidden_state1)
        elif track == 1:
            _, (hidden_state2, _) = self.bidlstm2(x)
            hidden_state2 = torch.flatten(torch.transpose(hidden_state2,
                                                          dim0=0,
                                                          dim1=1),
                                          start_dim=1)
            mu = self.mu2(hidden_state2)
            sigma = self.sigma2(hidden_state2)
        elif track == 2:
            _, (hidden_state3, _) = self.bidlstm3(x)
            hidden_state3 = torch.flatten(torch.transpose(hidden_state3,
                                                          dim0=0,
                                                          dim1=1),
                                          start_dim=1)
            mu = self.mu3(hidden_state2)
            sigma = self.sigma3(hidden_state2)
        else:
            raise ValueError("track must be 0/1/2, {} received".format(track))

        return (mu, sigma)

    def encode(self, x, control=[1, 1, 1]):
        """encode process demo, multi tracks input
        Args:
            x: A batch of sequences [sequence_length, batch_size, input_depth, 3]
               The number "3" indicates the fact that it's a 3-track input
               The 3 tracks MUST be ordered [lead, chord, drum]
        Returns:
            distribution: distribution of z torch.distributions.Normal
        """
        if not len(control) == 3:
            raise ValueError(
                "length of control must be 3, control received: {}".format(
                    control))
        """OUT OF DATE
        hidden_state = torch.cat((hidden_state1, hidden_state2, hidden_state3),
                                 dim=2)
        # hidden state [2, batch_size, 3*hidden_size]
        hidden_state = torch.flatten(
            torch.transpose(hidden_state, dim0=0, dim1=1))
        # hidden state [batch_size, 3*2*hidden_size]
        """
        track_num = 0
        for element in control:
            if element != 0:
                track_num += 1
        if track_num == 0:
            raise ValueError(
                "control indicates no track input, control received: {}".
                format(control))
        elif track_num != x.size()[-1]:
            raise ValueError(
                "control indicates {} tracks input, but x shape: {}".format(
                    track_num, x.size()))
        mu1, sigma1 = self.compose(x[:, :, :, 0],
                                   track=0) if control[0] else (0, 0)
        mu2, sigma2 = self.compose(x[:, :, :, 0],
                                   track=0) if control[0] else (0, 0)
        mu3, sigma3 = self.compose(x[:, :, :, 0],
                                   track=0) if control[0] else (0, 0)
        # This "mean" method is not reasonable at all!
        # But for compatness with 1 track input, we have no choice...
        mu = (mu1 + mu2 + mu3) / track_num
        sigma = (sigma1 + sigma2 + sigma3) / track_num

        # form distribution
        distribution = ds.Normal(loc=mu, scale=sigma)
        return distribution

    def __C_D_LOOP__(self, model, fc, conduct):
        """main loop from conduct to section output
        Args:
            model: model to use
            fc: fc to use
            conduct: conduct to use (scalar)
        Returns:
            output: section output [sequence_length, batch_size, input_depth]
        """
        # The first note of the first section
        control = torch.squeeze(conduct[0], dim=0)
        section_input = torch.cat(
            (torch.zeros([1, control.size()[1], self.input_depth]), control),
            dim=-1)
        _, (hidden_state, cell_state) = model(section_input)
        new_note = torch.unsqueeze(fc(hidden_state), dim=0)
        output = new_note
        # The rest of the first section
        for j in range(self.section_length - 1):
            section_input = torch.cat(new_note, control)
            _, (hidden_state, cell_state) = model(section_input,
                                                  (hidden_state, cell_state))
            new_note = torch.unsqueeze(fc(hidden_state), dim=0)
            output = torch.cat((output, new_note), dim=0)
        # The rest sections
        for i in range(1, self.repeat):
            control = torch.squeeze(conduct[i], dim=0)
            # No more special first note needed
            for j in range(self.section_length):
                section_input = torch.cat(new_note, control)
                _, (hidden_state,
                    cell_state) = model(section_input,
                                        (hidden_state, cell_state))
                new_note = torch.unsqueeze(fc(hidden_state), dim=0)
                output = torch.cat((output, new_note), dim=0)
        return output

    def forward(self, x, control=[1, 1, 1]):
        """Forward through VAE net
        Args:
            x: A batch of sequences [sequence_length, batch_size, input_depth, 3]
        Returns:
            distribution: distribution of z torch.distributions.Normal
            output: [sequence_length, batch_size, input_depth, 3]
        """
        distribution = self.encode(x, control)
        # sample from the distribution
        z = distribution.sample()
        # z [batch_size, z_size]
        # conduct
        z = self.fc1(z)
        # z [batch_size, control_depth]
        z = torch.unsqueeze(z, dim=0)
        _, (hidden_state, cell_state) = self.conductor(z)
        z = self.fc2(hidden_state)
        conduct = torch.unsqueeze(z, dim=0)
        # Enter loop
        for i in range(self.repeat - 1):
            _, (hidden_state,
                cell_state) = self.conductor(z, (hidden_state, cell_state))
            z = self.fc2(hidden_state)
            conduct = torch.cat((conduct, torch.unsqueeze(z, dim=0)), dim=0)
        # decode
        for i in range(self.repeat):
            sequence1 = self.__C_D_LOOP__(self.lstm1, self.fc4, conduct)
            sequence2 = self.__C_D_LOOP__(self.lstm2, self.fc5, conduct)
            sequence3 = self.__C_D_LOOP__(self.lstm3, self.fc6, conduct)
            sequence = torch.stack((sequence1, sequence2, sequence3), dim=3)
        return (distribution, sequence)

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

    def train_batch(self, inputs):
        """training batch process demo
        Args:
            inputs: [sequence_length, batch_size, input_depth, 3]
        Returns:
            None
        """
        self.train()

        distribution, outputs = self.forward(inputs)
        loss = self.loss_fn(inputs, outputs, distribution)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1

        return loss

    def train_inputs(self,
                     inputs,
                     epoch_num=10,
                     batch_size=600,
                     save=None,
                     checkpoint=None,
                     print_frequency=1):
        """training process demo
        Args:
            inputs: [total_size, sequence_length, input_depth, 3]
        Returns:
            None
        """
        self.step = 0
        shape = (int(inputs.size()[0] / batch_size), inputs.size()[1],
                 batch_size, inputs.size()[2], 3)
        inputs = torch.transpose(inputs, dim0=0, dim1=1)
        batches = torch.split(inputs, batch_size, dim=1)
        assert inputs.size() == shape
        for epoch in range(epoch_num):
            for batch in batches:
                loss = self.train_batch(batch)
            if epoch % print_frequency == 0:
                print("epoch {}, loss {}".format(epoch, loss))
