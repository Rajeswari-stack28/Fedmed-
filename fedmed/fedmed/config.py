from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
KEYS = ROOT / "keys"
CERTS = ROOT / "certs"

NUM_NODES = 3
NODE_NAMES = ["hospital-A", "hospital-B", "hospital-C"]
SPATIAL = (64, 64, 64)          # volumes are resized to this for CPU-friendly training
SERVER_ADDR = "localhost:8080"  # Flower gRPC endpoint
WS_HOST, WS_PORT = "localhost", 8765  # dashboard metrics stream
