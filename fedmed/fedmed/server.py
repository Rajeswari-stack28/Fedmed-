"""FedMed aggregator. Sees only ciphertext (HE mode); streams round metrics to the dashboard."""
import argparse
import json
import time

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from . import dp, he
from .config import ARTIFACTS, CERTS, KEYS, NODE_NAMES, NUM_NODES, SERVER_ADDR, WS_HOST, WS_PORT
from .hub import MetricsHub
from .model import build_model, get_weights


class FedMedStrategy(FedAvg):
    def __init__(self, hub, use_he, expected_nodes, num_rounds, ctx=None, **kw):
        super().__init__(**kw)
        self.hub, self.use_he, self.expected, self.num_rounds, self.ctx = hub, use_he, expected_nodes, num_rounds, ctx
        self.history = []

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round == 1:  # start only when every hospital has joined; later rounds tolerate dropouts
            client_manager.wait_for(self.expected, timeout=600)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if self.use_he:
            blobs = [parameters_to_ndarrays(r.parameters) for _, r in results]
            total_n = sum(r.num_examples for _, r in results)
            agg = he.add_ciphertexts(self.ctx, blobs)  # server never decrypts
            params = ndarrays_to_parameters(agg + [np.array([total_n], dtype=np.float64)])
        else:
            params, _ = super().aggregate_fit(server_round, results, failures)
        nodes = [{"node": r.metrics["node"], "train_loss": r.metrics["train_loss"], "n": r.num_examples,
                  "upload_mb": r.metrics["upload_mb"], "update_norm": r.metrics["update_norm"],
                  "secs": r.metrics["secs"]} for _, r in results]
        sigma = max(r.metrics["dp_noise"] for _, r in results)
        self.hub.publish({
            "type": "fit", "round": server_round, "nodes": nodes,
            "dropped": self.expected - len(results),
            "payload": "ciphertext (CKKS)" if self.use_he else "plaintext weights",
            "dp_noise": sigma,
            "epsilon": dp.approx_epsilon(sigma, server_round) if sigma > 0 else None,
        })
        return params, {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        n = sum(r.num_examples for _, r in results)
        wavg = lambda f: sum(f(r) * r.num_examples for _, r in results) / n
        ev = {"type": "eval", "round": server_round, "loss": wavg(lambda r: r.loss),
              "dice": wavg(lambda r: r.metrics["dice"]), "tc": wavg(lambda r: r.metrics["dice_tc"]),
              "wt": wavg(lambda r: r.metrics["dice_wt"]), "et": wavg(lambda r: r.metrics["dice_et"]),
              "per_node": {r.metrics["node"]: r.metrics["dice"] for _, r in results}}
        self.hub.publish(ev)
        self.history.append({**ev, "ts": time.time()})
        ARTIFACTS.mkdir(exist_ok=True)
        (ARTIFACTS / "history.json").write_text(json.dumps({"he": self.use_he, "rounds": self.history}, indent=2))
        return ev["loss"], {"dice": ev["dice"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--address", default=SERVER_ADDR)
    p.add_argument("--he", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-nodes", type=int, default=2, help="minimum nodes needed to run a round")
    args = p.parse_args()

    ctx = None
    if args.he:
        ctx = he.load_ctx(KEYS / "server_context.bin")
        assert ctx.is_public(), "server context must NOT contain the secret key"

    init = get_weights(build_model())
    hub = MetricsHub(WS_HOST, WS_PORT)
    baseline_path = ARTIFACTS / "baseline.json"
    hub.publish({"type": "config", "rounds": args.rounds, "he": args.he, "tls": args.tls, "nodes": NUM_NODES,
                 "node_names": NODE_NAMES, "params": int(sum(w.size for w in init)),
                 "baseline": json.loads(baseline_path.read_text()) if baseline_path.exists() else None})

    strategy = FedMedStrategy(
        hub, args.he, NUM_NODES, args.rounds, ctx=ctx,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=args.min_nodes, min_evaluate_clients=args.min_nodes, min_available_clients=args.min_nodes,
        initial_parameters=ndarrays_to_parameters(init),   # round 1 is plaintext: a random init carries no patient data
        on_fit_config_fn=lambda r: {"server_round": r, "encrypted": args.he and r > 1},
        on_evaluate_config_fn=lambda r: {"server_round": r, "encrypted": args.he},
    )
    certs = tuple(open(CERTS / f, "rb").read() for f in ("ca.crt", "server.pem", "server.key")) if args.tls else None
    fl.server.start_server(server_address=args.address, config=fl.server.ServerConfig(num_rounds=args.rounds),
                           strategy=strategy, certificates=certs)


if __name__ == "__main__":
    main()
