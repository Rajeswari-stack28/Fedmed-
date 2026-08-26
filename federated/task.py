import torch

from centralized.model import Simple3DUNet


def get_model():
    """Create a new 3D U-Net model."""
    return Simple3DUNet()


def get_parameters(model):
    """Convert model parameters to NumPy arrays."""
    return [
        value.detach().cpu().numpy()
        for _, value in model.state_dict().items()
    ]


def set_parameters(model, parameters):
    """Load NumPy parameters into the model."""

    state_dict = model.state_dict()

    new_state_dict = {}

    for (key, old_value), new_value in zip(
        state_dict.items(),
        parameters
    ):
        new_state_dict[key] = torch.tensor(
            new_value,
            dtype=old_value.dtype
        )

    model.load_state_dict(new_state_dict)

    return model