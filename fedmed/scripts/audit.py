"""Mid-project 'Federated Audit': does the federated model approach the centralized baseline?"""
import json
import sys
from pathlib import Path

A = Path(__file__).resolve().parent.parent / "artifacts"
if not (A / "history.json").exists() or not (A / "baseline.json").exists():
    sys.exit("Need artifacts/history.json (run the federation) and artifacts/baseline.json (run the baseline).")
hist, base = json.loads((A / "history.json").read_text()), json.loads((A / "baseline.json").read_text())
rounds = hist["rounds"]
print(f"{'round':>5} {'val_loss':>9} {'dice':>7}   per-hospital dice")
for r in rounds:
    print(f"{r['round']:>5} {r['loss']:>9.4f} {r['dice']:>7.4f}   " + "  ".join(f"{k}={v:.3f}" for k, v in r["per_node"].items()))
final, ratio = rounds[-1]["dice"], rounds[-1]["dice"] / max(base["dice"], 1e-9)
improving = rounds[-1]["loss"] < rounds[0]["loss"]
print(f"\ncentralized baseline dice : {base['dice']:.4f}\nfederated final dice      : {final:.4f}  ({ratio:.1%} of baseline)")
print(f"converging (loss down)    : {'PASS' if improving else 'FAIL'}")
print(f">=95% of baseline         : {'PASS' if ratio >= 0.95 else 'FAIL'}")
print(f"server payload            : {'ciphertext only' if hist['he'] else 'PLAINTEXT weights (HE disabled)'}")
print("raw data sent to server   : none (clients upload weights only; see fedmed/client.py fit())")
