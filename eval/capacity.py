"""X9 — Courbe de capacité (protocole pré-enregistré, EXTENSIONS.md §X9).

E1-multi varié à N ∈ {5, 10, 20, 40, 80} faits injectés dans la même M (contextes
tous distincts, pool.py), d fixe, config par défaut. Pour chaque N :
  - Δlogp moyen de rappel (baseline M=0 calculée une fois pour les 80 paires) ;
  - keysim max par fait (cos max de ses clés vs toutes les clés déjà écrites) ;
  - corrélation Δlogp / keysim max (re-test du prédicteur par fait — réserve I1 :
    corr trouvée POSITIVE intra-régime, le prédicteur part défavori).
Falaise (critère pré-enregistré) : premier N dont le Δlogp moyen tombe sous 50 %
de la référence N=5.

Usage :
  python eval/capacity.py
  python eval/capacity.py --model HuggingFaceTB/SmolLM2-360M --layer 16 --cap 0.1
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from pool import SECRETS_80, fact_pairs  # noqa: E402
from engram import EngramConfig, EngramEngine  # noqa: E402

SIZES = [5, 10, 20, 40, 80]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--cap", type=float, default=None)
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer
    if args.cap is not None:
        cfg.max_read_norm = args.cap
    cfg.track_keys = True

    print(f"[X9 capacity] {cfg.summary()}")
    engine = EngramEngine(cfg)
    pairs = fact_pairs(80)

    # Baseline M=0, une fois pour les 80 paires (déterministe, indépendante de N).
    engine.reset_memory()
    base = []
    for (_, question), secret in zip(pairs, SECRETS_80):
        engine.clear_context()
        base.append(engine.logprob_continuation(question, f" {secret}")[0])

    ref_mean = None
    pooled: list[tuple[float, float]] = []
    for n in SIZES:
        engine.reset_memory()
        fact_keysim = []
        for i in range(n):
            engine.clear_context()
            start = len(engine.memory.write_similarities)
            engine.stream(pairs[i][0].format(secret=SECRETS_80[i]), force_write=True)
            sims = engine.memory.write_similarities[start:]
            fact_keysim.append(max(sims) if sims else float("nan"))

        deltas = []
        for i in range(n):
            engine.clear_context()
            lp, _ = engine.logprob_continuation(pairs[i][1], f" {SECRETS_80[i]}")
            deltas.append(lp - base[i])

        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas)
        pos = sum(d > 0 for d in deltas)
        valid = [(d, k) for d, k in zip(deltas, fact_keysim) if k == k]
        corr = statistics.correlation(*zip(*valid)) if len(valid) >= 3 else float("nan")
        pooled.extend(valid)
        if ref_mean is None:
            ref_mean = mean
        cliff = "  << FALAISE (< 50 % de N=5)" if mean < 0.5 * ref_mean else ""
        print(
            f"  N={n:<3} Δlogp={mean:+.3f} ± {sd:.3f}  ({pos}/{n} > 0)  "
            f"keysim max: méd {statistics.median(fact_keysim):.3f} / max {max(fact_keysim):.3f}  "
            f"corr(Δ, keysim)={corr:+.2f}{cliff}"
        )

    print(f"\n  corr(Δlogp, keysim max) POOLÉE (tous N) : {statistics.correlation(*zip(*pooled)):+.2f}")
    print(f"  référence falaise : 50 % de N=5 = {0.5 * ref_mean:+.3f}")


if __name__ == "__main__":
    main()
