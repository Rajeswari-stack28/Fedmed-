"""Export one validation slice + ground truth + prediction from the final global model for the dashboard."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedmed.config import ARTIFACTS, NUM_NODES
from fedmed.data import build_dataset, get_partition
from fedmed.model import build_model

synthetic = "--synthetic" in sys.argv
root = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--data-root"), "data/BraTS2021")
_, val = get_partition(root, synthetic, NUM_NODES)[0]
sample = build_dataset(val[:1], synthetic)[0]
model = build_model()
model.load_state_dict(torch.load(ARTIFACTS / "global_node0.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    pred = (torch.sigmoid(model(sample["image"][None])) > 0.5)[0].numpy()
truth = sample["label"].numpy() > 0.5
z = int(np.argmax(truth[1].sum((0, 1))))  # axial slice with the largest whole-tumour area


def lab(m):  # 0 bg, 1 WT, 2 TC, 3 ET (innermost wins)
    o = np.zeros(m.shape[1:3], dtype=int)
    o[m[1][:, :, z]] = 1
    o[m[0][:, :, z]] = 2
    o[m[2][:, :, z]] = 3
    return o.tolist()


img = sample["image"][0, :, :, z].numpy()
img = (img - img.min()) / (img.max() - img.min() + 1e-9)
out = Path(__file__).resolve().parent.parent / "dashboard" / "public" / "sample.json"
out.write_text(json.dumps({"size": list(img.shape), "image": np.round(img, 3).tolist(),
                           "truth": lab(truth), "pred": lab(pred)}))
print("wrote", out)
