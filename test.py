import torch
import VAE.demo as VAE


def networkConnectionTest():
    sequence_length = 48
    batch_size = 600
    input_depth = 3
    single_input = torch.zeros([sequence_length, batch_size, input_depth])
    multi_input = torch.zeros([sequence_length, batch_size, input_depth, 3])
    model = VAE.VAENet(input_depth)

    model.eval()

    distribution, outputs = model.forward(multi_input, control=[1, 1, 1])
    assert isinstance(
        distribution,
        torch.distributions.Normal), "distribution class error, got {}".format(
            distribution.__class__.__name__)
    assert outputs.size() == (
        sequence_length, batch_size, input_depth,
        3), "wrong output size for multi_input, got {}".format(outputs.size())
    z = distribution.sample()
    assert z.size() == (batch_size), "wrong output size for z, got {}".format(
        z.size())

    outputs = model.generate(single_input, 16)
    assert outputs.size() == (
        sequence_length, batch_size, input_depth,
        3), "wrong output size for single input, got {}".format(outputs.size())

    print("networkConnectionTest pass")


if __name__ == "__main__":
    networkConnectionTest()
    pass
