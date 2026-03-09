#!/usr/bin/env python3
"""
download_weights.py  (v3 — correct Conv1D weight layout)
---------------------------------------------------------
KEY FIX: HuggingFace GPT-2 uses Conv1D (not nn.Linear).
Conv1D stores weight as (in_features, out_features), so the forward
pass is:  y = x @ W + b   (NO transpose)

Our C code does:  acc += x[i] * W[i][j]  which is also  x @ W

Therefore: save W DIRECTLY with NO .T  (previous versions incorrectly
applied .T which gave W^T, causing x @ W^T != x @ W → garbage output)

Weight shapes (in_features × out_features):
  c_attn.weight : (768, 2304)   → C: c_attn_w[768][2304]
  c_attn c_proj : (768, 768)    → C: c_proj_w[768][768]
  mlp c_fc      : (768, 3072)   → C: c_fc_w[768][3072]
  mlp c_proj    : (3072, 768)   → C: c_proj_w[3072][768]

Usage:
  pip install transformers torch numpy
  python3 download_weights.py
"""

import struct, os, sys
import numpy as np

def save(f, arr):
    """Write array as row-major float32 little-endian."""
    f.write(np.asarray(arr, dtype='<f4').tobytes())

def main():
    try:
        from transformers import GPT2Model
    except ImportError:
        print("Run: pip install transformers torch numpy"); sys.exit(1)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "weights", "gpt2.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Loading GPT-2 (124M) …")
    model = GPT2Model.from_pretrained("gpt2")
    sd    = {k: v.detach().cpu().float().numpy()
             for k, v in model.state_dict().items()}
    cfg   = model.config

    L = cfg.n_layer; D = cfg.n_embd; V = cfg.vocab_size; S = cfg.n_positions
    FF = 4 * D
    print(f"  layers={L}, d_model={D}, d_ff={FF}, vocab={V}, max_seq={S}")

    # Verify Conv1D shapes (in, out) — not (out, in) like nn.Linear
    w0 = sd["h.0.attn.c_attn.weight"]
    assert w0.shape == (D, 3*D), f"Unexpected c_attn shape: {w0.shape}"
    print(f"  c_attn.weight shape: {w0.shape}  ✓ (in={D}, out={3*D})")

    print(f"Writing → {out_path}")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", 0x47505432, L))

        for l in range(L):
            p = f"h.{l}"

            # ── Attention (Conv1D: shape already (in, out) = correct for C) ──
            save(f, sd[f"{p}.attn.c_attn.weight"])    # (768, 2304) — NO .T
            save(f, sd[f"{p}.attn.c_attn.bias"])      # (2304,)
            save(f, sd[f"{p}.attn.c_proj.weight"])    # (768, 768)  — NO .T
            save(f, sd[f"{p}.attn.c_proj.bias"])      # (768,)

            # ── MLP (Conv1D: same logic) ──────────────────────────────────────
            save(f, sd[f"{p}.mlp.c_fc.weight"])       # (768, 3072) — NO .T
            save(f, sd[f"{p}.mlp.c_fc.bias"])         # (3072,)
            save(f, sd[f"{p}.mlp.c_proj.weight"])     # (3072, 768) — NO .T
            save(f, sd[f"{p}.mlp.c_proj.bias"])       # (768,)

            print(f"  layer {l:2d} ✓")

        # ── Embeddings ───────────────────────────────────────────────────────
        save(f, sd["wte.weight"])    # (50257, 768)
        save(f, sd["wpe.weight"])    # (1024, 768)

        # ── Layer norms ──────────────────────────────────────────────────────
        for l in range(L):
            save(f, sd[f"h.{l}.ln_1.weight"])   # (768,)
            save(f, sd[f"h.{l}.ln_1.bias"])     # (768,)
            save(f, sd[f"h.{l}.ln_2.weight"])   # (768,)
            save(f, sd[f"h.{l}.ln_2.bias"])     # (768,)

        save(f, sd["ln_f.weight"])   # (768,)
        save(f, sd["ln_f.bias"])     # (768,)

    mb = os.path.getsize(out_path) / 1024**2
    print(f"\nDone. {out_path}  ({mb:.1f} MB)")
    print("Next: make clean && make && ./llm_attn \"Hi how are you?\"")

if __name__ == "__main__":
    main()