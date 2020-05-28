from VAE.model import VAENet


def train(model, loader, epoch_num=10, batch_size=600, save=None):
    """Helper function for training
    Args:
        model: VAENet
        loader: loader defined in data.py
    Returns:
        None
    """
    assert isinstance(
        model,
        VAENet), "VAE train method requires a VAE network, got {}".format(
            model.__class__.__name__)
    inputs, targets = loader.getBatches_1()
    if len(inputs) > batch_size:
        inputs = inputs[:batch_size]
    if len(targets) > batch_size:
        targets = targets[:batch_size]
    model.train_inputs(inputs,
                       targets,
                       control=[1, 0, 0],
                       epoch_num=epoch_num,
                       save=save)


def compose(model, section):
    """Helper function for composing
    Args:
        model: VAENet
        section: heusristic [section_length, input_depth]
    Returns:
        output: [section_length*repeat, input_depth]
    """
    assert isinstance(
        model,
        VAENet), "VAE compose method requires a VAE network, got {}".format(
            model.__class__.__name__)
    assert section.size() == (model.section_length, model.input_depth)
    section = section.unsqueeze(dim=1).unsqueeze(dim=-1)
    _, output = model.forward(section, control=[1, 0, 0])
    output = output.squeeze()
    return output
