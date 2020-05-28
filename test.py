import torch
import VAE.demo as VAE


def networkConnectionTest():
    sequence_length = 96
    batch_size = 30
    input_depth = 3
    single_input = torch.zeros([sequence_length, batch_size, input_depth, 1])
    multi_input = torch.zeros([sequence_length, batch_size, input_depth, 3])
    model = VAE.VAENet(input_depth, z_size=256)

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
    assert z.size() == (batch_size,
                        256), "wrong output size for z, got {}".format(
                            z.size())

    distribution, outputs = model.forward(single_input, control=[1, 0, 0])
    assert isinstance(
        distribution,
        torch.distributions.Normal), "distribution class error, got {}".format(
            distribution.__class__.__name__)
    assert outputs.size() == (
        sequence_length, batch_size, input_depth,
        3), "wrong output size for single_input, got {}".format(outputs.size())
    z = distribution.sample()
    assert z.size() == (batch_size,
                        256), "wrong output size for z, got {}".format(
                            z.size())

    print("networkConnectionTest pass")


def networkTrainTest():
    sequence_length = 96
    total_size = 300
    batch_size = 30
    input_depth = 3
    epoch = 10
    inputs = torch.zeros([total_size, sequence_length, input_depth, 3])
    model = VAE.VAENet(input_depth, z_size=256)
    model.train_inputs(inputs,
                       epoch_num=epoch,
                       batch_size=batch_size,
                       print_frequency=1)
    print("networkTrainTest pass")


def networkSaveLoadTest():
    sequence_length = 96
    total_size = 300
    input_depth = 3
    inputs = torch.zeros([total_size, sequence_length, input_depth, 3])
    model = VAE.VAENet(input_depth, section_length=6, z_size=128)
    model.train_inputs(inputs, save="save.model")
    origin_parameters = model.parameters()
    model = torch.load("save.model")
    new_parameters = model.parameters()
    assert model.section_length == 6, "section length not equal, got {}".format(
        model.section_length)
    assert model.section_length == 6, "z_size not equal, got {}".format(
        model.z_size)
    assert new_parameters == origin_parameters, "parameters not equal, old {}, new {}".format(
        origin_parameters, new_parameters)
    print("networkSaveLoadTest pass")


if __name__ == "__main__":
    networkConnectionTest()
    networkTrainTest()
    networkSaveLoadTest()
    pass
