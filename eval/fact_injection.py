"""E1 — Injection de fait : M retient-elle un fait après vidage du cache KV ?

Protocole (docs/ARCHITECTURE.md §5, E1) :
  pour chaque secret : reset M → streamer le fait avec write forcé → clear_context
  (le cache KV meurt, M survit) → mesurer log P(secret | question) avec M active,
  puis la même chose après reset de M (même prompt exact — c'est le contrôle D7).

Métriques : Δlog-prob moyen ± écart-type, rang du 1er token du secret, writes/run.
Un Δ > 0 systématique est le signal minimal ; secret en top-10 serait un vrai résultat.

Usage :
  python eval/fact_injection.py                    # défauts (GPT-2 124M)
  python eval/fact_injection.py --layer 9 --lam 1.0 --eta 0.1
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 vs Δ/±

from engram import EngramConfig, EngramEngine  # noqa: E402

# Mots rares/concrets pour minimiser le prior du cortex sur la réponse.
SECRETS = [
    "swordfish", "obsidian", "marmalade", "zeppelin", "catapult",
    "lighthouse", "porcupine", "avalanche", "tambourine", "nebula",
]
FACT_TEMPLATE = "The password is {secret}. Remember it well."

# E1 = rappel sur indice exact ; E1b = rappel par indice PARTIEL (paraphrases).
# Si exact ≫ paraphrase, la mémoire fait du par-cœur, pas de l'association —
# c'est le symptôme déclencheur de X2 (CA3) et un mauvais présage pour E2.
QUESTIONS = [
    ("exact", "The password is"),
    ("para1", "The secret word is"),
    ("para2", "Remember, the password was"),
    ("para3", "Enter the password:"),
]


def run_one(engine: EngramEngine, secret: str) -> dict:
    fact = FACT_TEMPLATE.format(secret=secret)

    # Phase d'injection : write forcé (on mesure la capacité de M, pas le gating).
    engine.reset_memory()
    engine.clear_context()
    records = engine.stream(fact, force_write=True)
    writes = sum(r.wrote for r in records)

    # Rappel avec M active — le cache KV est vidé : ce qui reste n'est pas du
    # contexte. La lecture ne contamine pas M (write=False dans logprob_continuation).
    mem = {}
    for name, question in QUESTIONS:
        engine.clear_context()
        mem[name] = engine.logprob_continuation(question, f" {secret}")

    # Contrôle : mêmes prompts exacts, M remise à zéro.
    engine.reset_memory()
    base = {}
    for name, question in QUESTIONS:
        engine.clear_context()
        base[name] = engine.logprob_continuation(question, f" {secret}")

    return {
        "secret": secret,
        "writes": writes,
        "deltas": {name: mem[name][0] - base[name][0] for name, _ in QUESTIONS},
        "rank_mem": mem["exact"][1],
        "rank_base": base["exact"][1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--secrets", type=int, default=len(SECRETS), help="nombre de secrets testés")
    parser.add_argument("--dg-dim", type=int, default=None, help="X1 : dimension gyrus denté (0 = off)")
    parser.add_argument("--dg-topk", type=int, default=None)
    parser.add_argument("--cap", type=float, default=None, help="max_read_norm (fraction de ‖h‖)")
    parser.add_argument("--hebbian", action="store_true", help="ablation D5 : Hebb pur (sans terme correctif)")
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer
    if args.lam is not None:
        cfg.lam = args.lam
    if args.eta is not None:
        cfg.eta = args.eta
    if args.dg_dim is not None:
        cfg.dg_dim = args.dg_dim
    if args.dg_topk is not None:
        cfg.dg_topk = args.dg_topk
    if args.cap is not None:
        cfg.max_read_norm = args.cap
    if args.hebbian:
        cfg.hebbian_only = True

    print(f"[E1 fact injection] {cfg.summary()}")
    engine = EngramEngine(cfg)

    results = [run_one(engine, s) for s in SECRETS[: args.secrets]]
    for r in results:
        per_q = "  ".join(f"{name}={r['deltas'][name]:+.3f}" for name, _ in QUESTIONS)
        print(
            f"  {r['secret']:<12} {per_q}  "
            f"rang exact: {r['rank_base']} → {r['rank_mem']}  writes={r['writes']}"
        )

    print()
    for name, question in QUESTIONS:
        deltas = [r["deltas"][name] for r in results]
        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        print(f'  Δlog-prob moyen [{name}] "{question}" : {mean:+.3f} ± {sd:.3f} nats (N={len(deltas)})')

    exact_mean = statistics.mean(r["deltas"]["exact"] for r in results)
    para_means = [
        statistics.mean(r["deltas"][name] for r in results)
        for name, _ in QUESTIONS
        if name != "exact"
    ]
    para_mean = statistics.mean(para_means)
    ratio = para_mean / exact_mean if exact_mean else float("nan")
    top10 = sum(r["rank_mem"] <= 10 for r in results)
    print(f"\n  généralisation (moy. paraphrases / exact) : {para_mean:+.3f} / {exact_mean:+.3f} = {ratio:.2f}")
    print(f"  secrets en top-10 avec M (question exacte) : {top10}/{len(results)}")
    if all(r["writes"] == 0 for r in results):
        print("  !! 0 write partout : run INVALIDE (voir pièges §4.3), pas un résultat négatif.")


if __name__ == "__main__":
    main()
