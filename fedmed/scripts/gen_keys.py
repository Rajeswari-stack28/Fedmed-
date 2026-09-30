"""Generate CKKS contexts. client_context.bin (has secret key) -> hospitals only. server_context.bin -> server."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedmed import he
from fedmed.config import KEYS

KEYS.mkdir(exist_ok=True)
secret, public = he.make_contexts()
(KEYS / "client_context.bin").write_bytes(secret)
(KEYS / "server_context.bin").write_bytes(public)
os.chmod(KEYS / "client_context.bin", 0o600)
print("Wrote keys/client_context.bin (SECRET, hospitals only) and keys/server_context.bin (public)")
