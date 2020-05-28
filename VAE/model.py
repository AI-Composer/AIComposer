import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds

from tqdm import tqdm
import math


class VAENet(nn.Module):
    """A really simplified demo of VAE net"""
    def __init__(self,
                 input_depth,
                 section_length=12,
                 encoder_size=1024,
                 encoder_layers=1,
                 decoder_size=512,
                 decoder_layers=1,
                 z_size=256,
                 conductor_size=512,
                 control_depth=256,
                 free_bits=0,
                 beta_rate=0,
                 max_beta=1,
                 lr=0.001):
        # TODO ADD support for layers
        super(VAENet, self).__init__()
        self.input_depth = input_depth
        self.section_length = section_length
        self.z_size = z_size
        self.free_bits = free_bits
        self.beta_rate = beta_rate
        self.max_beta = max_beta

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
        mu2, sigma2 = self.compose(x[:, :, :, 1],
                                   track=1) if control[1] else (0, 0)
        mu3, sigma3 = self.compose(x[:, :, :, 2],
                                   track=2) if control[2] else (0, 0)
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
        control = torch.unsqueeze(conduct[0], dim=0)
        section_input = torch.cat(
            (torch.zeros([1, control.size()[1], self.input_depth]), control),
            dim=-1)
        _, (hidden_state, cell_state) = model(section_input)
        new_note = fc(hidden_state)
        output = new_note
        # The rest of the first section
        for j in range(self.section_length - 1):
            section_input = torch.cat((new_note, control), dim=-1)
            _, (hidden_state, cell_state) = model(section_input,
                                                  (hidden_state, cell_state))
            new_note = fc(hidden_state)
            output = torch.cat((output, new_note), dim=0)
        # The rest sections
        for i in range(1, self.repeat):
            control = torch.unsqueeze(conduct[i], dim=0)
            # No more special first note needed
            for j in range(self.section_length):
                section_input = torch.cat((new_note, control), dim=-1)
                _, (hidden_state,
                    cell_state) = model(section_input,
                                        (hidden_state, cell_state))
                new_note = fc(hidden_state)
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
        conduct = z
        # Enter loop
        for i in range(self.repeat - 1):
            _, (hidden_state,
                cell_state) = self.conductor(z, (hidden_state, cell_state))
            z = self.fc2(hidden_state)
            conduct = torch.cat((conduct, z), dim=0)
        # decode
        sequence1 = self.__C_D_LOOP__(self.lstm1, self.fc4, conduct)
        sequence2 = self.__C_D_LOOP__(self.lstm2, self.fc5, conduct)
        sequence3 = self.__C_D_LOOP__(self.lstm3, self.fc6, conduct)
        sequence = torch.stack((sequence1, sequence2, sequence3), dim=3)
        return (distribution, sequence)

    def loss_fn(self, outputs, distribution, targets):
        """music VAE loss function, complicated, mainly KL_divergence
        Args:
            outputs: [sequence_length, batch_size, input_depth, 3]
            distribution: torch.distributions.Normal
            targets: [sequence_length, batch_size, 3, 3]
        Returns:
            loss: scalar
            r_cost: scalar
            beta: scalar
            kl_cost: scalar
        """
        q_z = distribution
        p_z = ds.Normal(loc=torch.zeros([self.z_size]),
                        scale=torch.ones([self.z_size]))
        kl_div = ds.kl_divergence(q_z, p_z)
        # allow some free bits
        free_nats = self.free_bits * math.log(2.0)
        # this cost limit the encoder to approach a standard distribution for z
        kl_cost = torch.mean(
            torch.max(input=(kl_div - free_nats), other=torch.zeros([1])))

        # musicVAE decoder loss function is too complicated, simplified here
        # FIXME Hard coding here
        pitch_out = torch.flatten(torch.transpose(outputs[:, :, :29],
                                                  dim0=2,
                                                  dim1=3),
                                  start_dim=0,
                                  end_dim=-2)
        duration_out = torch.flatten(torch.transpose(outputs[:, :, 29:41],
                                                     dim0=2,
                                                     dim1=3),
                                     start_dim=0,
                                     end_dim=-2)
        volumn_out = outputs[:, :, 41]
        pitch_target = torch.flatten(targets[:, :, 0]).to(torch.long)
        duration_target = torch.flatten(targets[:, :, 1]).to(torch.long)
        volumn_target = targets[:, :, 2].to(torch.float32)

        pitch_cost = F.cross_entropy(pitch_out, pitch_target)
        duration_cost = F.cross_entropy(duration_out, duration_target)
        volumn_cost = F.mse_loss(volumn_out, volumn_target)

        r_cost = (pitch_cost + duration_cost + volumn_cost) / 3

        # Now use a proper weight to balance between the two losses
        beta = ((1.0 - math.pow(self.beta_rate, float(self.step))) *
                self.max_beta)
        r_cost = torch.mean(r_cost)
        kl_cost = torch.mean(kl_cost)
        loss = r_cost + beta * kl_cost
        return (loss, r_cost, beta, kl_cost)

    def train_batch(self, inputs, targets, control=[1, 1, 1]):
        """training batch process demo
        Args:
            inputs: [sequence_length, batch_size, input_depth, track_num]
            targets: [sequence_length, batch_size, 3, track_num]
        Returns:
            None
        """
        self.train()
        self.repeat = int(inputs.size()[0] / self.section_length)

        distribution, outputs = self.forward(inputs, control)

        new_targets = []
        idx = 0
        for switch in control:
            if switch:
                new_targets.append(targets[:, :, :, idx])
                idx += 1
            else:
                new_targets.append(torch.zeros(targets.size()[:3]))

        targets = torch.stack(new_targets, dim=3)
        assert targets.size()[3] == 3

        loss, r_cost, beta, kl_cost = self.loss_fn(outputs, distribution,
                                                   targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1

        return (loss, r_cost, beta, kl_cost)

    def train_inputs(self,
                     inputs,
                     targets,
                     control=[1, 1, 1],
                     epoch_num=10,
                     save=None,
                     print_frequency=1):
        """training process demo
        Args:
            inputs: [batch_num, sequence_length, batch_size, input_depth, track_num]
            targets: [batch_num, sequence_length, batch_size, 3, track_num]
        Returns:
            None
        """
        self.step = 0
        print(
            "Training process start!\n",
            "input shape {}, target shape {}, control {}, epoch_num {}, save to {}, print_frequency {}\n"
            .format(inputs[0].size(), targets[0].size(), control, epoch_num,
                    save, print_frequency),
            "There are {} batches".format(len(inputs)))
        for epoch in range(epoch_num):
            for batch, target in tqdm(zip(inputs, targets)):
                assert isinstance(
                    batch, torch.Tensor), "input is not Tensor but {}".format(
                        batch.__class__.__name__)
                assert isinstance(
                    target,
                    torch.Tensor), "target is not Tensor but {}".format(
                        target.__class__.__name__)
                assert len(
                    batch.size()) == 4, "wrong batch shape, got {}".format(
                        batch.size())
                assert len(
                    target.size()) == 4, "wrong target shape, got {}".format(
                        target.size())
                loss, r_cost, beta, kl_cost = self.train_batch(
                    batch, target, control)
                print("epoch {}, loss: {}, r_cost: {}, beta: {}, kl_cost: {}".
                      format(epoch, loss, r_cost, beta, kl_cost))
            if epoch % print_frequency == 0:
                print("epoch {}, loss: {}, r_cost: {}, beta: {}, kl_cost: {}".
                      format(epoch, loss, r_cost, beta, kl_cost))
        if save is not None:
            assert isinstance(save, str), "save path must be string"
            torch.save(self, save)
            print("model saved to {}".format(save))
