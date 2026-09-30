"""FedMed hospital node. Raw MRI never leaves this process: only (optionally DP-noised, encrypted) weights do."""
import argparse
import os
import time

import flwr as fl
import numpy as np
import torch
from monai.data import DataLoader

from . import dp, he
from .config import ARTIFACTS, CERTS, KEYS, NODE_NAMES, NUM_NODES, SERVER_ADDR
from .data import build_dataset, get_partition
from .engine import evaluate, make_loss, train_epoch
from .model import build_model, get_weights, set_weights


class FedMedClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.node_id = args.node_id
        self.name = NODE_NAMES[self.node_id]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_items, val_items = get_partition(
            args.data_root, args.synthetic, NUM_NODES, args.seed, limit=args.limit, n_synth=args.n_synth)[self.node_id]
        self.train_ds = build_dataset(train_items, args.synthetic)
        self.train_loader = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(build_dataset(val_items, args.synthetic), batch_size=1)
        self.model = build_model().to(self.device)
        self.loss_fn = make_loss()
        self.shapes = [w.shape for w in get_weights(self.model)]
        self.numel = int(sum(np.prod(s) for s in self.shapes))
        self.ctx = he.load_ctx(KEYS / "client_context.bin") if args.he else None
        self.rng = np.random.default_rng(1000 + self.node_id)
        print(f"[{self.name}] train={len(self.train_ds)} val={len(self.val_loader.dataset)} "
              f"params={self.numel} he={args.he} dp_clip={args.dp_clip} dp_noise={args.dp_noise}")

    # Flower requires this, but the server supplies initial parameters so it is never called.
    def get_parameters(self, config):
        return get_weights(self.model)

    def _load_global(self, parameters, encrypted):
        if encrypted:
            total_n = float(parameters[-1][0])
            flat = he.decrypt(self.ctx, parameters[:-1], self.numel) / total_n
            set_weights(self.model, he.unflatten(flat, self.shapes))
        else:
            set_weights(self.model, parameters)

    def fit(self, parameters, config):
        rnd, t0 = int(config["server_round"]), time.time()
        self._load_global(parameters, bool(config["encrypted"]))
        global_flat = he.flatten(get_weights(self.model))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

        def maybe_crash(step):  # simulates a hospital going offline mid-epoch (mid-project review demo)
            if self.args.crash_at_round == rnd and step >= 2:
                print(f"[{self.name}] SIMULATED CRASH in round {rnd}", flush=True)
                os._exit(1)

        loss = 0.0
        for _ in range(self.args.local_epochs):
            loss = train_epoch(self.model, self.train_loader, self.loss_fn, opt, self.device, on_step=maybe_crash)

        local_flat = he.flatten(get_weights(self.model))
        upd_flat, norm = dp.privatize(local_flat, global_flat, self.args.dp_clip, self.args.dp_noise, self.rng)
        n = len(self.train_ds)
        if self.args.he:
            payload = he.encrypt(self.ctx, upd_flat * n)          # Enc(n_i * w_i)
        else:
            payload = he.unflatten(upd_flat, self.shapes)
        mb = sum(a.nbytes for a in payload) / 1e6
        metrics = {"node": self.name, "train_loss": float(loss), "update_norm": norm, "upload_mb": mb,
                   "dp_noise": float(self.args.dp_noise), "dp_clip": float(self.args.dp_clip), "secs": time.time() - t0}
        print(f"[{self.name}] round {rnd}: loss={loss:.4f} upload={mb:.1f}MB", flush=True)
        return payload, n, metrics

    def evaluate(self, parameters, config):
        self._load_global(parameters, bool(config["encrypted"]))
        loss, per = evaluate(self.model, self.val_loader, self.loss_fn, self.device)
        ARTIFACTS.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), ARTIFACTS / f"global_node{self.node_id}.pt")
        m = {"node": self.name, "dice": float(per.mean()), "dice_tc": float(per[0]),
             "dice_wt": float(per[1]), "dice_et": float(per[2])}
        return float(loss), len(self.val_loader.dataset), m


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--node-id", type=int, required=True, choices=range(NUM_NODES))
    p.add_argument("--server", default=SERVER_ADDR)
    p.add_argument("--tls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ca-cert", default=str(CERTS / "ca.crt"))
    p.add_argument("--he", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-synth", type=int, default=16, help="synthetic volumes per hospital")
    p.add_argument("--data-root", default="data/BraTS2021")
    p.add_argument("--limit", type=int, default=None, help="cap total subjects (quick runs)")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dp-clip", type=float, default=0.0, help="L2 clip of update; 0 disables DP")
    p.add_argument("--dp-noise", type=float, default=0.0, help="noise multiplier sigma")
    p.add_argument("--crash-at-round", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse()
    torch.manual_seed(args.seed)
    client = FedMedClient(args)
    root = open(args.ca_cert, "rb").read() if args.tls else None
    fl.client.start_client(server_address=args.server, client=client.to_client(), root_certificates=root)


if __name__ == "__main__":
    main()
