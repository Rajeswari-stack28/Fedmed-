"""Homomorphic encryption helpers (TenSEAL / CKKS).

Trust model
  * Hospitals share ONE context that contains the secret key (distributed out-of-band, see scripts/gen_keys.py).
  * The server only ever holds the PUBLIC context: it can add ciphertexts, never decrypt them.
Protocol
  client -> server : Enc(n_i * w_i)     (n_i = local training examples)
  server           : sum_i Enc(n_i * w_i), plus plaintext scalar N = sum_i n_i
  server -> client : Enc(sum) + [N];   client decrypts and divides by N  => weighted FedAvg, dropout-safe.
"""
from pathlib import Path

import numpy as np
import tenseal as ts

POLY_DEGREE = 8192
SLOTS = POLY_DEGREE // 2


def make_contexts():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=POLY_DEGREE, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    secret = ctx.serialize(save_secret_key=True)
    ctx.make_context_public()
    return secret, ctx.serialize()


def load_ctx(path):
    return ts.context_from(Path(path).read_bytes())


def flatten(arrays):
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def unflatten(flat, shapes):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(flat[i:i + n].reshape(s).astype(np.float32))
        i += n
    return out


def encrypt(ctx, flat):
    return [np.frombuffer(ts.ckks_vector(ctx, flat[i:i + SLOTS].tolist()).serialize(), dtype=np.uint8).copy()
            for i in range(0, len(flat), SLOTS)]


def decrypt(ctx, blobs, numel):
    out = []
    for b in blobs:
        out.extend(ts.ckks_vector_from(ctx, b.tobytes()).decrypt())
    return np.asarray(out[:numel], dtype=np.float64)


def add_ciphertexts(ctx, list_of_blob_lists):
    acc = None
    for blobs in list_of_blob_lists:
        cts = [ts.ckks_vector_from(ctx, b.tobytes()) for b in blobs]
        acc = cts if acc is None else [x + y for x, y in zip(acc, cts)]
    return [np.frombuffer(c.serialize(), dtype=np.uint8).copy() for c in acc]
