"""Shared train / eval loops (used by federated clients AND the centralized baseline)."""
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric


def make_loss():
    return DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)


def train_epoch(model, loader, loss_fn, opt, device, on_step=None):
    model.train()
    total, steps = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item()
        steps += 1
        if on_step:
            on_step(steps)
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Returns (mean loss, per-channel dice ndarray [TC, WT, ET])."""
    model.eval()
    dice = DiceMetric(include_background=True, reduction="mean_batch")
    total, n = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x)
        total += loss_fn(logits, y).item() * x.shape[0]
        n += x.shape[0]
        dice(y_pred=(torch.sigmoid(logits) > 0.5).float(), y=y)
    per = torch.nan_to_num(dice.aggregate()).cpu().numpy()
    return total / max(n, 1), per
