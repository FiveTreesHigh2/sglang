"""Bitwise equivalence check for GDN chunk_h BV tile variants.

BV only re-partitions the V dimension of chunk_gated_delta_rule_fwd_h; the
reduction order of every output element (K-dim 64-slices, BT loop) does not
depend on BV, so BV=32 (default) and BV=64 (tuned for SM120) must produce
bitwise-identical o / h / final state. Any difference beyond torch.equal
indicates a boundary/indexing bug in the tile, not float noise.

BV is baked into the triton autotune config at module import time from
SGLANG_GDN_CHUNK_H_BV, so each variant must run in its own subprocess.

Usage (run on a CUDA machine):
    python test_gdn_chunk_h_bv_equivalence.py                 # compare 32 vs 64
    python test_gdn_chunk_h_bv_equivalence.py --bv-a 32 --bv-b 64
"""

import argparse
import os
import subprocess
import sys
import tempfile

# Production-like GDN shape (Qwen3.5-35B-A3B): H=32 v-heads, K=V=128.
H, K, V = 32, 128, 128
POOL_SIZE = 64
# Varlen lengths covering sub-chunk, exact-chunk, non-multiple and long
# sequences (chunk size 64), ~8.2k total tokens like one prefill chunk.
LENGTHS = [1, 7, 63, 64, 65, 130, 257, 511, 1023, 2048, 4031]
SEED = 20260722


def _build_inputs(device):
    import torch

    torch.manual_seed(SEED)
    total = sum(LENGTHS)
    n = len(LENGTHS)

    cu_seqlens = torch.zeros(n + 1, dtype=torch.long, device=device)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(LENGTHS, dtype=torch.long, device=device), dim=0
    )
    indices = torch.randperm(POOL_SIZE, device=device)[:n].to(torch.int32)
    pool = torch.randn(POOL_SIZE, H, V, K, dtype=torch.float32, device=device) * 0.1

    dtype = torch.bfloat16
    q = torch.randn(1, total, H, K, dtype=dtype, device=device)
    k = torch.randn(1, total, H, K, dtype=dtype, device=device)
    v = torch.randn(1, total, H, V, dtype=dtype, device=device)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, total, H, dtype=dtype, device=device)
    )
    beta = torch.sigmoid(torch.randn(1, total, H, dtype=dtype, device=device))
    return q, k, v, g, beta, pool, indices, cu_seqlens


def _worker(out_path):
    import torch

    # The fla package lives at kernels.ops.attention in this repo but at
    # srt.layers.attention in the venv-installed build on the serving host.
    try:
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
        from sglang.kernels.ops.attention.fla.chunk_delta_h import GDN_CHUNK_H_BV
    except ImportError:
        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
        from sglang.srt.layers.attention.fla.chunk_delta_h import GDN_CHUNK_H_BV

    device = "cuda"
    q, k, v, g, beta, pool, indices, cu_seqlens = _build_inputs(device)

    # h below is the raw fwd_h output; pool slots hold the in-place final state.
    o, _, h = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=pool,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    torch.save(
        {
            "bv": GDN_CHUNK_H_BV,
            "o": o.cpu(),
            "h": h.cpu(),
            "final_state": pool[indices.long()].cpu(),
        },
        out_path,
    )


def _run_variant(bv, out_path):
    env = {
        **os.environ,
        "SGLANG_GDN_CHUNK_H_BV": str(bv),
        "SGLANG_GDN_CHUNK_H_NUM_WARPS": "4",
        "SGLANG_GDN_CHUNK_H_NUM_STAGES": "2",
    }
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", "--out", out_path],
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--bv-a", type=int, default=32)
    parser.add_argument("--bv-b", type=int, default=64)
    args = parser.parse_args()

    if args.worker:
        _worker(args.out)
        return

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, f"bv{args.bv_a}.pt")
        path_b = os.path.join(tmp, f"bv{args.bv_b}.pt")
        _run_variant(args.bv_a, path_a)
        _run_variant(args.bv_b, path_b)

        res_a = torch.load(path_a, weights_only=True)
        res_b = torch.load(path_b, weights_only=True)
        assert res_a["bv"] == args.bv_a and res_b["bv"] == args.bv_b, (
            f"env did not take effect: got BV {res_a['bv']} / {res_b['bv']}"
        )

        failed = False
        for key in ("o", "h", "final_state"):
            ta, tb = res_a[key].float(), res_b[key].float()
            if torch.equal(res_a[key], res_b[key]):
                print(f"[PASS] {key}: bitwise identical, shape={tuple(ta.shape)}")
            else:
                failed = True
                diff = (ta - tb).abs()
                mism = (res_a[key] != res_b[key]).sum().item()
                print(
                    f"[FAIL] {key}: {mism}/{ta.numel()} elements differ, "
                    f"max_abs_diff={diff.max().item():.3e}"
                )
        if failed:
            print(f"\nBV={args.bv_a} vs BV={args.bv_b}: NOT bitwise equivalent")
            sys.exit(1)
        print(f"\nBV={args.bv_a} vs BV={args.bv_b}: bitwise equivalent, OK")


if __name__ == "__main__":
    main()
