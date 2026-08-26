import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import Simple3DUNet
from dataset import SyntheticMRIDataset


def dice_score(prediction, target):

    prediction = (prediction > 0.5).float()

    intersection = (prediction * target).sum()

    dice = (
        (2.0 * intersection + 1e-8)
        /
        (prediction.sum() + target.sum() + 1e-8)
    )

    return dice.item()


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Create dataset
    dataset = SyntheticMRIDataset(num_samples=20)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False
    )

    # Create model
    model = Simple3DUNet().to(device)

    # Loss function
    loss_function = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5

    for epoch in range(epochs):

        # -------------------
        # TRAINING
        # -------------------

        model.train()

        total_loss = 0

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            predictions = model(images)

            loss = loss_function(
                predictions,
                masks
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        # -------------------
        # VALIDATION
        # -------------------

        model.eval()

        total_dice = 0

        with torch.no_grad():

            for images, masks in validation_loader:

                images = images.to(device)
                masks = masks.to(device)

                predictions = torch.sigmoid(
                    model(images)
                )

                total_dice += dice_score(
                    predictions,
                    masks
                )

        average_dice = (
            total_dice / len(validation_loader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {average_loss:.4f} "
            f"| Dice: {average_dice:.4f}"
        )

    # Save model
    torch.save(
        model.state_dict(),
        "centralized_baseline.pt"
    )

    print("\nModel saved as centralized_baseline.pt")


if __name__ == "__main__":
    main()