# FedMed — Cross-Silo Federated Learning Engine for Brain-Tumour Segmentation

Three "hospitals" train a 3D U-Net on private MRI data. Raw scans never leave a node. Only **DP-noised, CKKS-encrypted**
weight updates travel to the aggregator, which adds ciphertexts without ever being able to read them.

```
 hospital-A ─┐  Enc(nᵢ·wᵢ)                          ┌─ WebSocket :8765 ─► React/Recharts dashboard
 hospital-B ─┼──── gRPC + TLS (:8080) ────► server ─┤
 hospital-C ─┘  ◄── Enc(Σ nᵢ·wᵢ) + [N] ──   (public HE context only, cannot decrypt)
   local MRI · PyTorch/MONAI 3D U-Net · clip + Gaussian noise (DP) · TenSEAL encrypt/decrypt
```

| Module | Where |
|---|---|
| Federated loop (Flower FedAvg, dropout tolerant) | `fedmed/server.py`, `fedmed/client.py` |
| 3D U-Net, Dice loss/metric (MONAI) | `fedmed/model.py`, `fedmed/engine.py` |
| BraTS loader + synthetic generator | `fedmed/data.py` |
| Homomorphic encryption (TenSEAL CKKS) | `fedmed/he.py`, `scripts/gen_keys.py` |
| Differential privacy | `fedmed/dp.py` |
| gRPC + TLS | `scripts/gen_certs.sh`, Flower `certificates=` |
| Live metrics + dashboard | `fedmed/hub.py`, `dashboard/` |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10 or 3.11 (TenSEAL wheels)
pip install -r requirements.txt
bash scripts/gen_certs.sh && python scripts/gen_keys.py
cd dashboard && npm install && cd ..
```

## Quick demo (no dataset needed)
```bash
./run_local.sh                      # server + 3 nodes on synthetic tumours, 5 rounds, TLS + HE on
cd dashboard && npm run dev         # open http://localhost:5173  (start it before or during the run)
```

## Real data (BraTS 2021)
Download BraTS 2021 Task 1 (Kaggle / Synapse) so that each case is `data/BraTS2021/BraTS2021_00000/BraTS2021_00000_{flair,t1,t1ce,t2,seg}.nii.gz`.
```bash
python scripts/centralized_baseline.py --data-root data/BraTS2021 --limit 60 --epochs 20   # Week 1 baseline
ROUNDS=10 ./run_local.sh --data-root data/BraTS2021 --limit 60 --local-epochs 2
```

## Week-by-week
| Week | Run |
|---|---|
| 1 | `python scripts/centralized_baseline.py …` · `python -m fedmed.server` + 3× `python -m fedmed.client --node-id N` (3 processes; nodes are distinct clients, server on :8080) |
| 2 | FedAvg loop and TLS are already wired: `./run_local.sh`. Ablate with `--no-tls`, `--no-he` |
| Mid review | `python scripts/audit.py` (convergence vs baseline). Node dropout: run nodes manually and add `--crash-at-round 3` to one; the round finishes with the other two (dashboard shows "dropped: 1") |
| 3 | HE is on by default (`--no-he` on server and clients disables it). Dashboard streams live over WebSocket |
| 4 | Add DP to every client: `--dp-clip 1.0 --dp-noise 0.01`. Then `python scripts/export_sample.py --synthetic` (or `--data-root …`) for the mask viewer |

## Honest limitations (read before claiming compliance)
* **Shared secret key.** All hospitals share one CKKS secret key so they can decrypt the aggregate. A hospital that colluded with the server
  could decrypt another's update. Production fix: threshold / multi-key HE or secure aggregation (MPC).
* **HE ≠ SMPC.** The original brief says MPC; this implements HE. Both keep the server blind, via different mechanisms.
* **Differential privacy is client-level** and `approx_epsilon` is a *loose* bound. Use an RDP accountant (Opacus) for reportable ε. Noise costs Dice; tune `--dp-noise` and plot the tradeoff.
* **Round 1 is plaintext** (a random initial model, no patient information).
* **Cost.** ~0.5M parameters ≈ 120 ciphertexts ≈ 50 MB per node per round. Larger models need bigger `channels` budgets or selective-layer encryption.
* Volumes are resized to 64³ for CPU training; raise `SPATIAL` in `config.py` on a GPU.
* This is a research/education codebase, not a certified HIPAA/GDPR system.
