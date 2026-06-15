#!/usr/bin/env python3
from __future__ import annotations
import sys, os
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)

"""
Continual-Learning Experiment — RDMCA core hypothesis
=====================================================
Tests the paper's central, cheapest-to-validate claim:

    Sector-isolated consolidation over a frozen core retains old skills while
    learning new ones, where standard sequential fine-tuning catastrophically
    forgets — at equal compute.

Self-contained: synthetic LM "domains" (disjoint vocab sub-ranges with their
own sequential structure). No downloads. Runs in ~1-2 min on an M2 Max.

Protocol (standard continual learning):
  - K domains are learned strictly in sequence.
  - After finishing each domain we evaluate perplexity on *all* domains.
  - We report the final per-domain perplexity and Backward Transfer (BWT):
        BWT = mean_{i<K} ( ppl_i(after task i)  −  ppl_i(after last task) )
    More negative BWT = more forgetting. RDMCA should be ~0 by construction.

Methods compared:
  naive  — one model, all parameters updated on each domain in turn.
  ewc    — naive + Elastic Weight Consolidation (diagonal Fisher penalty).
  rdmca  — frozen core + one LoRA sector per domain; only that sector updates
           (oracle routing by domain id, to isolate the consolidation claim
           from router accuracy).

Usage:
  python models/cognition/experiments/continual_learning.py
  python models/cognition/experiments/continual_learning.py --domains 5 --steps 300 --d_model 128
"""
import argparse
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.core.model.transformer import RDMCAFoundational, ModelConfig
from src.core.model.lora import SectorAdapter, LoRAConfig, masked_sector_update


# ---------------------------------------------------------------------------
# Synthetic domains
# ---------------------------------------------------------------------------
def make_domain(domain_id: int, vocab_per_domain: int, seq_len: int,
                batch: int, rng: np.random.Generator) -> mx.array:
    """
    A batch of [batch, seq_len+1] sequences for one domain. All domains SHARE
    the same vocab range [4, 4+V); they differ only in the affine transition
    rule x' = (a_d·x + b_d) mod V. Shared vocab keeps perplexities finite and
    forgetting graded (not pathological), so baselines like EWC behave sensibly
    while the same parameters must encode conflicting rules — which is exactly
    where sector isolation should help.
    """
    V = vocab_per_domain
    base = 4                                          # reserve 0-3 for specials
    a = (1 + 2 * domain_id) % V or 1                  # odd → coprime with even V
    b = (7 * domain_id + 1) % V                       # domain-specific offset
    rows = []
    for _ in range(batch):
        x = int(rng.integers(0, V))
        seq = []
        for _ in range(seq_len + 1):
            seq.append(base + x)
            x = (a * x + b) % V                       # deterministic next state
        rows.append(seq)
    return mx.array(np.array(rows, dtype=np.int32))


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def eval_ppl(model, domain_id, vocab_per_domain, seq_len, rng,
             active_sector=None, n_batches=4) -> float:
    if active_sector is not None and model.sectors:
        model.set_active_sectors([(active_sector, 1.0)])
    elif model.sectors is not None:
        model.set_active_sectors([])
    losses = []
    for _ in range(n_batches):
        batch = make_domain(domain_id, vocab_per_domain, seq_len, 16, rng)
        loss = model.eval_ce(batch)
        mx.eval(loss)
        losses.append(loss.item())
    return float(np.exp(np.mean(losses)))


# ---------------------------------------------------------------------------
# EWC (diagonal Fisher) penalty
# ---------------------------------------------------------------------------
def diag_fisher(model, batches):
    """Diagonal Fisher estimate = mean grad^2 over batches (full-param)."""
    def loss_fn(m, t):
        return m.mrl_loss(t)
    lg = nn.value_and_grad(model, loss_fn)
    acc = None
    for b in batches:
        _, grads = lg(model, b)
        flat = dict(tree_flatten(grads))
        sq = {k: (v * v) for k, v in flat.items()}
        acc = sq if acc is None else {k: acc[k] + sq[k] for k in acc}
    return {k: v / len(batches) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Training methods
# ---------------------------------------------------------------------------
def small_model(d_model, vocab, seq_len) -> RDMCAFoundational:
    cfg = ModelConfig(vocab_size=vocab, d_model=d_model, n_layers=3,
                      n_heads=max(1, d_model // 64), ffn_dim=d_model * 4,
                      context_len=seq_len + 1,
                      mrl_dims=[d_model // 2, d_model])
    return RDMCAFoundational(cfg)


def run_naive(args, vocab, rng, ewc=False):
    model = small_model(args.d_model, vocab, args.seq_len)
    opt = optim.AdamW(learning_rate=args.lr)
    lg = nn.value_and_grad(model, lambda m, t: m.mrl_loss(t))

    fisher, star = None, None
    lam = args.ewc_lambda

    def loss_with_ewc(m, t):
        loss = m.mrl_loss(t)
        if ewc and fisher is not None:
            cur = dict(tree_flatten(m.trainable_parameters()))
            pen = mx.array(0.0)
            for k in fisher:
                if k in cur and k in star:
                    pen = pen + (fisher[k] * (cur[k] - star[k]) ** 2).sum()
            loss = loss + 0.5 * lam * pen
        return loss

    lg_ewc = nn.value_and_grad(model, loss_with_ewc)
    after_task = {}     # ppl_i right after learning task i

    for d in range(args.domains):
        for _ in range(args.steps):
            b = make_domain(d, args.vocab_per_domain, args.seq_len, args.batch, rng)
            _, grads = (lg_ewc if ewc else lg)(model, b)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
        after_task[d] = eval_ppl(model, d, args.vocab_per_domain, args.seq_len, rng)
        if ewc:
            bs = [make_domain(d, args.vocab_per_domain, args.seq_len, args.batch, rng)
                  for _ in range(4)]
            f = diag_fisher(model, bs)
            fisher = f if fisher is None else {k: fisher[k] + f[k] for k in f}
            star = dict(tree_flatten(model.trainable_parameters()))

    final = {d: eval_ppl(model, d, args.vocab_per_domain, args.seq_len, rng)
             for d in range(args.domains)}
    return after_task, final


def run_rdmca(args, vocab, rng):
    model = small_model(args.d_model, vocab, args.seq_len)
    # Brief generic warmup so the frozen core has usable features, then freeze.
    opt = optim.AdamW(learning_rate=args.lr)
    lg = nn.value_and_grad(model, lambda m, t: m.mrl_loss(t))
    for _ in range(args.warmup):
        d = int(rng.integers(0, args.domains))
        b = make_domain(d, args.vocab_per_domain, args.seq_len, args.batch, rng)
        _, grads = lg(model, b)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    # One LoRA sector per domain; attach and freeze the core.
    sectors = {d + 1: SectorAdapter(LoRAConfig(d_model=model.cfg.d_model,
                                               n_layers=model.cfg.n_layers,
                                               sector_id=d + 1, rank=args.rank))
               for d in range(args.domains)}
    model.attach_sectors(sectors)
    model.freeze()

    after_task = {}
    for d in range(args.domains):
        sid = d + 1
        opt_d = optim.AdamW(learning_rate=args.lr)
        for _ in range(args.steps):
            b = make_domain(d, args.vocab_per_domain, args.seq_len, args.batch, rng)
            def loss_fn(m, _b=b, _sid=sid):
                m.set_active_sectors([(_sid, 1.0)])
                return m.mrl_loss(_b)
            masked_sector_update(model, sid, loss_fn, opt_d)
        after_task[d] = eval_ppl(model, d, args.vocab_per_domain, args.seq_len,
                                 rng, active_sector=sid)

    final = {d: eval_ppl(model, d, args.vocab_per_domain, args.seq_len, rng,
                         active_sector=d + 1)
             for d in range(args.domains)}
    return after_task, final


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def bwt(after_task, final):
    """Backward transfer in ppl terms: negative = forgetting (ppl rose)."""
    deltas = [after_task[i] - final[i] for i in range(len(final) - 1)]
    return float(np.mean(deltas)) if deltas else 0.0


def report(name, after_task, final, elapsed):
    K = len(final)
    avg = float(np.mean([final[d] for d in range(K)]))
    b = bwt(after_task, final)
    row = " ".join(f"D{d}:{final[d]:6.1f}" for d in range(K))
    print(f"\n[{name}]  ({elapsed:.1f}s)")
    print(f"  final ppl per domain : {row}")
    print(f"  mean final ppl       : {avg:8.2f}   (lower = better)")
    print(f"  backward transfer    : {b:+8.2f}   (~0 good; very negative = forgetting)")
    return avg, b


def main():
    ap = argparse.ArgumentParser(description="RDMCA continual-learning experiment")
    ap.add_argument("--domains", type=int, default=5)
    ap.add_argument("--steps", type=int, default=250, help="train steps per domain")
    ap.add_argument("--warmup", type=int, default=150, help="rdmca core warmup steps")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--vocab_per_domain", type=int, default=64,
                    help="shared vocab size V across all domains")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ewc_lambda", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vocab = 4 + args.vocab_per_domain                 # shared vocab range
    print("=" * 64)
    print("RDMCA continual-learning experiment")
    print(f"  domains={args.domains} steps/domain={args.steps} d_model={args.d_model} "
          f"vocab={vocab} rank={args.rank}")
    print("  Sequential learning; eval on all domains after each. BWT≈0 is the goal.")
    print("=" * 64)

    results = {}
    for name, fn in [("naive", lambda: run_naive(args, vocab, np.random.default_rng(args.seed))),
                     ("ewc",   lambda: run_naive(args, vocab, np.random.default_rng(args.seed), ewc=True)),
                     ("rdmca", lambda: run_rdmca(args, vocab, np.random.default_rng(args.seed)))]:
        t0 = time.time()
        at, fin = fn()
        results[name] = report(name, at, fin, time.time() - t0)

    print("\n" + "=" * 64)
    print("SUMMARY (mean final perplexity | backward transfer)")
    for name in ("naive", "ewc", "rdmca"):
        avg, b = results[name]
        print(f"  {name:6s}  ppl={avg:8.2f}  BWT={b:+8.2f}")
    naive_bwt = results["naive"][1]
    rdmca_bwt = results["rdmca"][1]
    print("-" * 64)
    if rdmca_bwt > naive_bwt + 1e-6:
        print(f"  RDMCA forgets less than naive fine-tuning "
              f"(BWT {rdmca_bwt:+.2f} vs {naive_bwt:+.2f}). Hypothesis supported.")
    else:
        print("  No clear RDMCA advantage at these settings — try more domains/steps.")
    print("=" * 64)


if __name__ == "__main__":
    main()
