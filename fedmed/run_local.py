"""Cross-platform launcher: 1 server + 3 hospital nodes.
   python run_local.py                                  # synthetic data, 5 rounds
   python run_local.py --rounds 8 --data-root data/BraTS2021 --limit 60
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--rounds", type=int, default=5)
a, extra = p.parse_known_args()
if not extra:
    extra = ["--synthetic"]

if not (root / "certs" / "ca.crt").exists():
    subprocess.check_call([sys.executable, str(root / "scripts" / "gen_certs.py")])
if not (root / "keys" / "client_context.bin").exists():
    subprocess.check_call([sys.executable, str(root / "scripts" / "gen_keys.py")])

procs = [subprocess.Popen([sys.executable, "-m", "fedmed.server", "--rounds", str(a.rounds)], cwd=root)]
time.sleep(5)
procs += [subprocess.Popen([sys.executable, "-m", "fedmed.client", "--node-id", str(i), *extra], cwd=root) for i in range(3)]
try:
    for pr in procs:
        pr.wait()
except KeyboardInterrupt:
    pass
finally:
    for pr in procs:
        if pr.poll() is None:
            pr.terminate()