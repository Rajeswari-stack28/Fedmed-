"""Week 1: centralized 3D U-Net baseline on the UNION of all hospitals' data (the accuracy ceiling)."""
import argparse
import json
import sys
from pathlib import Path

import torch
from monai.data import DataLoader
from torch.utils.data import ConcatDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedmed.config import ARTIFACTS, NUM_NODES
from fedmed.data import build_dataset, get_partition
from fedmed.engine import evaluate, make_loss, train_epoch
from fedmed.model import build_model

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--synthetic", action="store_true")
p.add_argument("--n-synth", type=int, default=16)
p.add_argument("--data-root", default="data/BraTS2021")
p.add_argument("--limit", type=int, default=None)
p.add_argument("--batch-size", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

torch.manual_seed(a.seed)
parts = get_partition(a.data_root, a.synthetic, NUM_NODES, a.seed, limit=a.limit, n_synth=a.n_synth)
tr = ConcatDataset([build_dataset(t, a.synthetic) for t, _ in parts])
va = ConcatDataset([build_dataset(v, a.synthetic) for _, v in parts])
tl, vl = DataLoader(tr, batch_size=a.batch_size, shuffle=True), DataLoader(va, batch_size=1)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, loss_fn = build_model().to(dev), make_loss()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
for ep in range(1, a.epochs + 1):
    tloss = train_epoch(model, tl, loss_fn, opt, dev)
    vloss, per = evaluate(model, vl, loss_fn, dev)
    print(f"epoch {ep:>3}  train_loss={tloss:.4f}  val_loss={vloss:.4f}  dice={per.mean():.4f} (TC/WT/ET={per.round(3)})")

ARTIFACTS.mkdir(exist_ok=True)
(ARTIFACTS / "baseline.json").write_text(json.dumps({"dice": float(per.mean()), "loss": float(vloss), "epochs": a.epochs}))
print("saved artifacts/baseline.json")
